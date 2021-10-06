#!/bin/bash

#SBATCH --nodes=1 ### No change
#SBATCH --mem=36GB
#SBATCH --cpus-per-task=8 ### No less than gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=trts

module purge


singularity exec --nv \
     --overlay /scratch/yd1008/nyu-capstone/pytorch1.8.0-cuda11.1.ext3:ro \
    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
    /bin/bash -c "source /ext3/env.sh; python transformer.py"
ls