#!/bin/bash

#SBATCH --nodes=1 ### No change
#SBATCH --mem=36GB
#SBATCH --cpus-per-task=8 ### No less than gpu
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=trts
##gpu:2 -c2 or gpu:k80:3 -c3
module purge
set -e
OVERLAY_DIRECTORY=/scratch/work/public/overlay-fs-ext3/
if [[ ! -d $OVERLAY_DIRECTORY ]]; then
OVERLAY_DIRECTORY=/scratch/wz2247/singularity/overlays/
fi
TMP_OVERLAY_SOURCE=overlay-0.5GB-200K.ext3
TMPDIR=${TMPDIR:-/tmp}
TMP_OVERLAY=${TMP_OVERLAY:-overlay-temp.ext3}
IMAGE=${IMAGE:-/scratch/wz2247/singularity/images/pytorch_21.06-py3.sif}
if [[ ! -f $TMP_OVERLAY ]]; then
echo "Temporary overlay not found, automatically creating a new one."
cp "$OVERLAY_DIRECTORY/$TMP_OVERLAY_SOURCE.gz" "$TMPDIR"
gunzip "$TMPDIR/$TMP_OVERLAY_SOURCE.gz"
mv "$TMPDIR/$TMP_OVERLAY_SOURCE" "$TMP_OVERLAY"
fi

singularity exec --no-home -B $HOME/.ssh -B /scratch -B $PWD --nv \
    --cleanenv \
    --overlay $TMP_OVERLAY \
    --overlay overlay-base.ext3:ro \
    --overlay overlay-packages.ext3:ro \
    $IMAGE /bin/bash -c "source ~/.bashrc; conda activate /ext3/conda/bootcamp; python train_transformer.py"
