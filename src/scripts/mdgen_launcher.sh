#!/bin/bash

# This script does NOT itself use SBATCH directives.
# Instead, it calls 'sbatch' on your existing mdgen_finetune.sh,
# passing in the different beta_start values you want.

# If you need 5 distinct values of beta_start, define them here:
beta_values=(5.0 1.0 0.1 0.01 0.001)

for beta in "${beta_values[@]}"; do
  echo "Submitting job with beta_start=${beta}"
  sbatch mdgen_finetune.sh "${beta}"
done