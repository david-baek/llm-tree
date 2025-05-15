## Generated using GPT, and then reviewed and modified by the user

## https://github.com/clovaai/donut/issues/57
import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import sys
import os
from huggingface_hub import InferenceClient,login
import copy
import random

from pathlib import Path
import argparse

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

token = ""
login(token=token)

from utils.model_utils import load_model_and_tokenizer, set_seed
seed = 81
set_seed(seed)
data_seed = 42

current_path = Path(__file__)
sys.path.append(str(current_path.parent.parent))

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir) #sys.path.append(cur_dir)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model argument.')
    parser.add_argument('--model_a', type=str, required=True, help='Specify Model A')
    parser.add_argument('--model_b', type=str, required=True, help='Specify Model B')
    parser.add_argument('--cut_depth', type=float, required=True, help='Specify Seed')
    args = parser.parse_args()
    model_a = args.model_a
    model_b = args.model_b
    cut_depth = args.cut_depth



import wandb
with open('../API_KEY', 'r') as file:
    api_key = file.read().strip() 
wandb.login(key=api_key)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_map = {
    "pythia70m": "EleutherAI/pythia-70m-deduped",
    "pythia160m": "EleutherAI/pythia-160m-deduped",
    "pythia410m": "EleutherAI/pythia-410m-deduped",
    "pythia1.4b": "EleutherAI/pythia-1.4b-deduped",
    "pythia2.8b": "EleutherAI/pythia-2.8b-deduped",
    "pythia6.9b": "EleutherAI/pythia-6.9b-deduped",
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "opt350m": "facebook/opt-350m",
    "opt1.3b": "facebook/opt-1.3b",
    "opt2.7b": "facebook/opt-2.7b",
    "opt6.7b": "facebook/opt-6.7b",
}

for key in model_name_map.keys():
    if key in model_a:
        model_name1 = model_name_map[key]
    if key in model_b:
        model_name2 = model_name_map[key]


cache_dir = "/om2/user/dbaek/MODELS"


tokenizer1 = AutoTokenizer.from_pretrained(model_name1, cache_dir=cache_dir, trust_remote_code=True)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2, cache_dir=cache_dir, trust_remote_code=True)

if tokenizer1.pad_token_id is None:
    tokenizer1.pad_token_id = tokenizer1.eos_token_id

if tokenizer2.pad_token_id is None:
    tokenizer2.pad_token_id = tokenizer2.eos_token_id

model1 = AutoModelForCausalLM.from_pretrained(model_name1,cache_dir=cache_dir,torch_dtype=torch.float32)
model2 = AutoModelForCausalLM.from_pretrained(model_name2,cache_dir=cache_dir, torch_dtype=torch.float32)


for param in model1.parameters():
    param.requires_grad = False
for param in model2.parameters():
    param.requires_grad = False

print(model1)
print(model2)
sys.stdout.flush()

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(CustomLinear, self).__init__(in_features, out_features, bias, dtype=torch.float32)

    def forward(self, x, **kwargs):
        x = super(CustomLinear, self).forward(x)
        ret = (x,)
        if 'output_attentions' in kwargs:
            ret += (None,)
        if 'use_cache' in kwargs:
            ret += (None,)
        return ret

stitching_layer = CustomLinear(model2.config.hidden_size, model1.config.hidden_size)
torch.nn.init.zeros_(stitching_layer.weight)
torch.nn.init.zeros_(stitching_layer.bias)
stitching_layer = stitching_layer.to(torch.float32)
stitching_layer.to(device)



if "pythia" in model_a:
    layer_num1 = int(cut_depth * len(model1.gpt_neox.layers))
    layer_num2 = int(cut_depth * len(model2.gpt_neox.layers))
    model1.gpt_neox.layers = nn.ModuleList(model2.gpt_neox.layers[:layer_num2] + [stitching_layer] + model1.gpt_neox.layers[layer_num1:])
    model1.gpt_neox.embed_in = model2.gpt_neox.embed_in
elif "opt" in model_a:
    layer_num1 = int(cut_depth * len(model1.model.decoder.layers))
    layer_num2 = int(cut_depth * len(model2.model.decoder.layers))
    model1.model.decoder.layers = nn.ModuleList(model2.model.decoder.layers[:layer_num2] + [stitching_layer] + model1.model.decoder.layers[layer_num1:])
    model1.model.decoder.embed_tokens = model2.model.decoder.embed_tokens
    model1.model.decoder.embed_positions = model2.model.decoder.embed_positions

model1.config.num_hidden_layers = len(model1.gpt_neox.layers) if "pythia" in model_a else len(model1.transformer.h) if "gpt" in model_a else len(model1.model.decoder.layers) if "opt" in model_a else 0
model1 = model1.to(device)
print(model1)

wandb.init(project="stitching-2", name=f"{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}")


'''
Prepare Dataset
'''

dataset_name = "monology/pile-uncopyrighted"
ctx_len = 512

train_data = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
eval_data = load_dataset("monology/pile-test-val", split="validation", streaming=True)
test_data = load_dataset("monology/pile-test-val", split="test",streaming=True)

train_data = train_data.shuffle(seed=data_seed,buffer_size=20000)
eval_data = eval_data.shuffle(seed=data_seed,buffer_size=20000)
test_data = test_data.shuffle(seed=data_seed,buffer_size=20000)

sample_eval_data = eval_data.take(20000)
sample_test_data = test_data.take(20000)

def tokenize_function(example):
    tokens1 = tokenizer1(
        example["text"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=ctx_len
    )
    '''
    for i in range(100):
        print(tokens1['input_ids'][i].size())
        print(tokenizer1.decode(tokens1['input_ids'][i], skip_special_tokens=True))
        print("-"*100)
        sys.stdout.flush()
    '''
    return tokenizer1(
        example["text"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=ctx_len
    )

def filter_examples(example):
    tokens1 = tokenizer1.tokenize(
        example['text'],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=ctx_len
    )
    return len(tokens1) >= ctx_len


train_tokenized_datasets = train_data.filter(filter_examples).map(tokenize_function, batched=True)
val_tokenized_datasets = sample_eval_data.filter(filter_examples).map(tokenize_function, batched=True)
test_tokenized_datasets = sample_test_data.filter(filter_examples).map(tokenize_function, batched=True)


# Custom Trainer that only trains the new stitching layer
class StitchedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(device)

        if input_ids.dim() != 2:
            input_ids = input_ids.unsqueeze(0)
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or Inf detected in parameters of layer {name}")

#        print(tokenizer1.decode(input_ids[0], skip_special_tokens=True))
#        sys.stdout.flush()
                
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']
#        print(logits.size())

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer1.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(loss)
        sys.stdout.flush()

        return (loss, outputs) if return_outputs else loss


class TrainLossConvergenceCallback(TrainerCallback):
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        """
        Args:
            patience (int): Number of steps/epochs to wait for improvement.
            threshold (float): Minimum improvement in loss to continue training.
        """
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.num_bad_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Get the training loss from the state
        if len(state.log_history) < 5:
            return
        tmp_list = []
        for i in range(1,6):
            tmp_list.append(state.log_history[-i]["loss"])
        current_loss = sum(tmp_list) / len(tmp_list)
        
        
        # Check if the loss has improved by at least the threshold
        if self.best_loss - current_loss >= self.threshold:
            self.best_loss = current_loss
            self.num_bad_steps = 0  # Reset counter on improvement
        else:
            self.num_bad_steps += 1  # Increment if no improvement

        print(current_loss, self.num_bad_steps)
        sys.stdout.flush()

        # Stop training if patience is exceeded
        
        if self.num_bad_steps >= self.patience:
            print(f"Stopping training early due to no improvement in training loss after {self.patience} steps.")
            control.should_training_stop = True
        


training_args = TrainingArguments(
    output_dir=f"../results/{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}",
    eval_strategy = "no",
    save_strategy = "no",
    logging_steps = 1,
    learning_rate = 1e-3,
#    per_device_train_batch_size=1,
#    per_device_eval_batch_size=1,
    seed = seed,
    data_seed = data_seed,
    num_train_epochs=3,
    weight_decay=1e-4,
    warmup_steps=100,
    max_steps = 10000,
    run_name = f"{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}",
)

#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


## Initialize Trainer
trainer = StitchedTrainer(
    model=model1,
    args=training_args,
#    data_collator=data_collator,
    train_dataset = train_tokenized_datasets,
    eval_dataset = val_tokenized_datasets,
    callbacks=[TrainLossConvergenceCallback(patience=200, threshold=0.001)],
)


# Train the model (train only the stitching layer)
train_output = trainer.train()

final_loss = train_output.training_loss
print(train_output)
print(f"Final training loss: {final_loss}")
sys.stdout.flush()


torch.save(stitching_layer.state_dict(), f"../results/{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}/{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}_stitched_model.pth")

do_eval = True

if do_eval:
    with torch.no_grad():
        test_losses = []
        idx_cnt = 0
        for example in test_tokenized_datasets:
            test_loss_item = trainer.compute_loss(model1, example).item()
            test_losses.append(test_loss_item)
            print(test_loss_item)
            sys.stdout.flush()
            idx_cnt += 1
            if idx_cnt >= 2000:
                break
        with open(f"../results/{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}/{model_b}_{layer_num2}_to_{model_a}_{layer_num1}_seed{seed}_test_loss.csv","w") as file:
            for loss_val in test_losses:
                file.write(f"{loss_val}\n")
