#!/bin/bash
read -ep 'GPU: ' gpu_var
read -ep 'Epochs: ' epoch_var
for i in 1e-3 1e-4 1e-5
do
  for j in 4 8
  do
    for b in 1 1.1 1.2
    do
      for d in 4 8 16 32
      do
        python3 train.py --epochs=$epoch_var --batch_size=$j --learning_rate=$i --beta=$b --latent_dim=$d --gpu=$gpu_var
      done
    done
  done
done