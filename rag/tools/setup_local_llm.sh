#!/usr/bin/env bash
# Local model server for the RAG pipeline (run inside WSL2 Ubuntu).
#
# The pipeline itself stays on Windows — it drives Microsoft Office through COM
# to get true page numbers — so only the model server lives here, exposed over
# an OpenAI-compatible HTTP endpoint.
#
#   bash setup_local_llm.sh install     # one-off: pip, venv, vLLM
#   bash setup_local_llm.sh serve [MODEL]
#
set -euo pipefail

VENV="$HOME/.venvs/vllm"
# 24 GB of VRAM comfortably fits a 7B vision model in bf16 with a long context.
MODEL_DEFAULT="Qwen/Qwen2.5-VL-7B-Instruct"
PORT="${PORT:-8000}"

# Keep the weights on the Linux filesystem: the Windows C: drive is nearly full
# and /mnt/c is also much slower for many-small-file reads.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

need_space() {
    local avail
    avail=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
    if [ "$avail" -lt 60 ]; then
        echo "!! $HOME の空きが ${avail}GB しかありません。モデル配置に 60GB 以上を推奨します。"
    fi
}

case "${1:-}" in
install)
    need_space
    echo "== apt: python3-pip / python3-venv =="
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-pip python3-venv

    echo "== venv: $VENV =="
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install --upgrade pip wheel

    echo "== vLLM (CUDA ビルド) =="
    pip install vllm

    echo
    python -c "import torch, vllm; print('vllm', vllm.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    echo
    echo "完了。次: bash $0 serve"
    ;;

serve)
    MODEL="${2:-$MODEL_DEFAULT}"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "== serving $MODEL on 0.0.0.0:$PORT =="
    echo "   初回はモデルのダウンロード (十数GB) が走ります"
    exec vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{"image":2}' \
        --served-model-name "$MODEL" \
        --disable-log-requests
    ;;

*)
    cat <<EOF
使い方:
  bash $0 install          pip/venv/vLLM を導入 (初回のみ)
  bash $0 serve [MODEL]    OpenAI互換サーバを起動

既定モデル: $MODEL_DEFAULT
VRAM 24GB での選択肢:
  Qwen/Qwen2.5-VL-7B-Instruct          約16GB  余裕あり・高速   (推奨)
  Qwen/Qwen2.5-VL-32B-Instruct-AWQ     約20GB  高精度・要調整
EOF
    ;;
esac
