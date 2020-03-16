#!/bin/bash
for i in 1e-3 1e-4 1e-5
do
  for j in 4 8
  do
    for b in 1 1.1 1.2
    do
      for d in 4 8 16 32
      do
        python3 train.py --epochs=200 --batch_size=$j --learning_rate=$i --beta=$b --latent_dim=$d --gpu=0
      done
    done
  done
done