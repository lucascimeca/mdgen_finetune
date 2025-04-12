#!/usr/bin/env bash

methods=('tb')
betas=('1.' '1e6')
replay_buffers=('False')
vargrads=('True')

for method in "${methods[@]}"; do
  # Set learning rate depending on the method
  if [ "$method" == "rtb" ]; then
    lr="6e-4"
  else
    lr="1e-5"
  fi

  for beta in "${betas[@]}"; do
    for vargrad in "${vargrads[@]}"; do
        for replay_buffer in "${replay_buffers[@]}"; do
          echo "Submitting job with method=$method, lr=$lr, beta=$beta, replay_buffer=$replay_buffer"
          sbatch mdgen_finetune.sh "$method" "$lr" "$beta" "$vargrad" "$replay_buffer"
        done
      done
    done
done
