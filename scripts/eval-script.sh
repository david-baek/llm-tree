#!/bin/bash

for i in {0..4};
do
    python ../src/eval-model.py --model_idx $i --shuffle 1
    python ../src/eval-model.py --model_idx $i --shuffle 0
done

