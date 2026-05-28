#!/bin/bash
#SBATCH --job-name=story_serve
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/t07an25/llm_experiments/logs/%j_story.out
#SBATCH --error=/home/t07an25/llm_experiments/logs/%j_story.err
# NOTE: the orchestrator needs NO GPU — it only fans out to the other services.
# It is pinned to gpu02 so the chat app's hard-coded http://gpu02:8772 resolves.
# If the scheduler refuses a CPU-only job on the gpu partition, uncomment:
##SBATCH --gres=gpu:3g.20gb:1

source /etc/profile.d/z00_lmod.sh
source /etc/profile.d/z01_uoa_lmod2.sh
modset_conf 2025-26
module load miniconda3 gcc/14.2.0
source /opt/software/uoa/apps/miniconda3/etc/profile.d/conda.sh
conda activate ditto

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1

NODE=$(hostname -s)
PORT=8772
export PORT=$PORT

# Upstream services (all on gpu02). Override here if any move.
export GEMMA_URL=http://gpu02:8766
export FLUX_URL=http://gpu02:8768
export TTS_URL=http://gpu02:8770

# Render settings
export STORY_W=1280
export STORY_H=720
export STORY_FPS=30

echo "============================================================"
echo "  Visual Story Orchestrator  |  Node: ${NODE}  |  Port: ${PORT}"
echo "  Gemma:${GEMMA_URL}  Flux:${FLUX_URL}  TTS:${TTS_URL}"
echo "  Use /storyboard or /story <url|text> in the chat UI"
echo "============================================================"

mkdir -p /home/t07an25/llm_experiments/logs
cd /home/t07an25/llm_experiments/

uvicorn story_app:app \
  --host 0.0.0.0 \
  --port ${PORT} \
  --log-level info \
  --timeout-keep-alive 1800
