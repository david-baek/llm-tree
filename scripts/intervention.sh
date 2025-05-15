#!/bin/bash

for i in {0..4};
do
    python ../src/icl_rep.py --model_idx $i
done

python ../src/ablation.py --model_idx 0 --exp_type layer

for i in {0..4};
do
    python ../src/ablation.py --model_idx $i --exp_type model
done


