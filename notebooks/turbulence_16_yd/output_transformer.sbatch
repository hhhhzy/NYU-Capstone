#!/bin/bash

#SBATCH --nodes=1 ### No change
#SBATCH --mem=36GB
#SBATCH --cpus-per-task=8 ### No less than gpu
#SBATCH --time=2:00:00
##SBATCH --gres=gpu:1
#SBATCH --job-name=trts
##gpu:2 -c2 or gpu:k80:3 -c3

module purge
set -e
OVERLAY_DIRECTORY=/scratch/work/public/overlay-fs-ext3/
if [[ ! -d $OVERLAY_DIRECTORY ]]; then
OVERLAY_DIRECTORY=/scratch/yd1008/
fi
TMP_OVERLAY_SOURCE=overlay-15GB-500K.ext3
TMPDIR=${TMPDIR:-/tmp}
TMP_OVERLAY=${TMP_OVERLAY:-overlay-temp.ext3}
IMAGE=${IMAGE:-/scratch/wz2247/singularity/images/pytorch_21.06-py3.sif}
if [[ ! -f $TMP_OVERLAY ]]; then
cp "$OVERLAY_DIRECTORY/$TMP_OVERLAY_SOURCE.gz" "$TMPDIR"
gunzip "$TMPDIR/$TMP_OVERLAY_SOURCE.gz"
mv "$TMPDIR/$TMP_OVERLAY_SOURCE" "$TMP_OVERLAY"
fi

singularity exec --no-home -B $HOME/.ssh -B /scratch -B $PWD --nv \
    --cleanenv \
    --overlay $TMP_OVERLAY \
    --overlay /scratch/yd1008/nyu-capstone/overlay-base.ext3:ro \
    --overlay /scratch/yd1008/nyu-capstone/overlay-packages.ext3:ro \
    $IMAGE /bin/bash -c "source ~/.bashrc; conda activate /ext3/conda/bootcamp; python transformer_output.py"