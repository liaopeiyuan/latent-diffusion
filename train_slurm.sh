#!/bin/bash
#SBATCH -J ldm  # Job name
#SBATCH -o ldm%j.log  # Name of stdout output file (%j expands to jobId)
#SBATCH -e ldm%j.err  # Name of stderr output file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl289@cornell.edu
#SBATCH -N 1  # Total number of CPU nodes requested
#SBATCH -n 8  # Total number of CPU cores requrested
#SBATCH -t 120:00:00  # Run time (hh:mm:ss)
#SBATCH --mem=50000  # CPU Memory pool for all cores
#SBATCH --partition=gpu --gres=gpu:titanrtx:4
#SBATCH --get-user-env

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/artbench-impressionism-ldm-vq-4.yaml -t -l training-runs/ldms \
--gpus 0,1,2,3 -r training-runs/ldms/2022-06-14T14-25-22_artbench-impressionism-ldm-vq-4/