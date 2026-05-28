#!/bin/bash
#SBATCH --job-name=ditto_talk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/t07an25/llm_experiments/logs/%j_ditto_talk.out
#SBATCH --error=/home/t07an25/llm_experiments/logs/%j_ditto_talk.err

source /etc/profile.d/z00_lmod.sh
source /etc/profile.d/z01_uoa_lmod2.sh
modset_conf 2025-26
module load miniconda3 gcc/14.2.0
source /opt/software/uoa/apps/miniconda3/etc/profile.d/conda.sh
conda activate ditto

# GCC 14 lib64 must come first so soxr_ext finds CXXABI_1.3.15
GCC14_LIB64=/opt/software/uoa/spk/0.23/linux-centos7-x86_64/gcc-12.1.0/gcc-14.2.0-cxb6tfeuf6d4j23lziof7ktzvyhjnek4/lib64
CONDA_ENV_LIB=/home/t07an25/sharedscratch/.conda/envs/ditto/lib
SITE=${CONDA_ENV_LIB}/python3.10/site-packages
TORCH_LIB=${SITE}/torch/lib
# Collect all nvidia CUDA runtime lib paths (cudnn, cublas, cufft, etc.)
NVIDIA_LIBS=$(find ${SITE}/nvidia -maxdepth 3 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${GCC14_LIB64}:${CONDA_ENV_LIB}:${TORCH_LIB}:${NVIDIA_LIBS}${LD_LIBRARY_PATH}"
unset GCC14_LIB64 CONDA_ENV_LIB SITE TORCH_LIB NVIDIA_LIBS

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

MIG_UUID=$(nvidia-smi -L 2>/dev/null | grep -oP 'MIG-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' | head -1)
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES=$MIG_UUID
    echo "MIG UUID: $MIG_UUID"
fi

NODE=$(hostname -s)
PORT=8770

export PORT=$PORT
export HF_HOME=/scratch/users/t07an25/llm_experiments/hf_cache

export DITTO_REPO=/home/t07an25/llm_experiments/ditto-talkinghead
export DITTO_DATA_ROOT=${DITTO_REPO}/checkpoints/ditto_pytorch
export DITTO_CFG=${DITTO_REPO}/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl

# ── Face image (required) ──────────────────────────────────────────────────────
# SCP your face photo to the HPC before submitting this job:
#   scp -i ~/.ssh/macleod1_key face.png t07an25@macleod1.abdn.ac.uk:~/llm_experiments/face.png
export TALK_FACE_PATH=/home/t07an25/llm_experiments/face.png

# ── Voice reference (optional) ────────────────────────────────────────────────
# Uncomment to clone a specific voice from a ~10 s WAV clip:
export TALK_VOICE_PATH=/home/t07an25/llm_experiments/voice_ref.wav

echo "============================================================"
echo "  Ditto Talking Head  |  Node: ${NODE}  |  Port: ${PORT}"
echo "  Chat app: http://${NODE}:${PORT}"
echo "  Use /talk <text> in the chat UI"
echo "============================================================"

mkdir -p /home/t07an25/llm_experiments/logs

cd /home/t07an25/llm_experiments/
uvicorn ditto_talk_app:app \
  --host 0.0.0.0 \
  --port ${PORT} \
  --log-level info \
  --timeout-keep-alive 600
