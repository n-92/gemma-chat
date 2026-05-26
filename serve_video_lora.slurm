#!/bin/bash
#SBATCH --job-name=wan_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/t07an25/llm_experiments/logs/%j_videolora.out
#SBATCH --error=/home/t07an25/llm_experiments/logs/%j_videolora.err

# Note: --mem=48G because the 14B model (~28 GB bfloat16) lives in CPU RAM
# when sequential offload is active.  48 G gives headroom for LoRA weights
# (~300-600 MB each) and OS overhead.

source /etc/profile.d/z00_lmod.sh
source /etc/profile.d/z01_uoa_lmod2.sh
modset_conf 2025-26
module load miniconda3 gcc/14.2.0
source /opt/software/uoa/apps/miniconda3/etc/profile.d/conda.sh
conda activate rag_gemma4

TORCH_LIB=/home/t07an25/sharedscratch/.conda/envs/rag_gemma4/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
unset TORCH_LIB

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

echo "============================================================"
echo "  Wan2.2 14B + LoRA  |  Node: ${NODE}  |  Port: ${PORT}"
echo "  Available LoRAs: doggy  spoon  sfbehind  transition"
echo "  LoRA repo: lkzd7/WAN2.2_LoraSet_NSFW"
echo "  Chat: /videolora <lora_name> <prompt>"
echo "  Direct: http://${NODE}:${PORT}"
echo "============================================================"

mkdir -p /home/t07an25/llm_experiments/logs

cd /home/t07an25/llm_experiments/
uvicorn video_lora_app:app \
  --host 0.0.0.0 \
  --port ${PORT} \
  --log-level info \
  --timeout-keep-alive 1800
