# nyu-capstone

## Usuage

### Set up the environment for GCP instance container

For first time usage:

1. Clone this repo to GCP instance.

2. In the main directory of nyu-bootcamp, run `./scripts/create_base_overlay.sh` and `./scripts/create_package_overlay.sh`.

Start from here if you have done step 1 and 2 before:

3. Start singularity with `./start_singularity.sh`

4. Run `conda activate /ext3/conda/bootcamp` to activate conda environment inside the singularity.

5. Run `./scripts/run_sshd.sh`
