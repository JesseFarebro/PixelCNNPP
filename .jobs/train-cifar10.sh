#!/bin/bash -l
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=00-10:00
#SBATCH --output=train-cifar10-%j.out
set -e; pushd ${SLURM_SUBMIT_DIR}
trap "{ popd; date; }" EXIT
pwd; hostname; nvidia-smi

module restore tf2; source ~/.venv/tf2/bin/activate

tensorboard --logdir=${SLURM_SUBMIT_DIR}/logdir --host 0.0.0.0 &
python3 main.py --config experiments/cifar10.gin multigpu
