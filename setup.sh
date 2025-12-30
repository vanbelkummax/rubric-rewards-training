#!/usr/bin/env bash
# Setup script for rubric rewards training
# Author: Max Van Belkum
# Date: 2025-12-30
# Version: 0.2.0 - Fixed model naming, added Unsloth

set -e  # Exit on error

echo "======================================================"
echo "Rubric Rewards Training Setup (v0.2.0)"
echo "======================================================"
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
python_version=$(python3 --version | cut -d' ' -f2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python 3.10+ required, found $python_version"
    exit 1
fi
echo "✓ Python $python_version"

# Check CUDA/GPU
echo ""
echo "[2/7] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "✓ GPU detected: $gpu_name ($vram_total MB VRAM)"

    if [ "$vram_total" -lt 20000 ]; then
        echo "WARNING: <20GB VRAM detected. Unsloth optimizations enabled but may still struggle."
    fi
else
    echo "ERROR: No GPU detected. This project requires CUDA GPU."
    exit 1
fi

# Create virtual environment
echo ""
echo "[3/7] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/7] Installing Python dependencies..."
echo "(This includes Unsloth - may take 10-15 minutes)"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check Ollama
echo ""
echo "[5/7] Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama not found. Install from https://ollama.ai"
    exit 1
fi

# CRITICAL FIX: Check for correct model (Qwen2.5, not Qwen3)
echo ""
echo "[6/7] Checking/pulling Qwen2.5-Coder-32B model..."
if ollama list | grep -q "qwen2.5-coder:32b"; then
    echo "✓ qwen2.5-coder:32b model found"
else
    echo "Pulling qwen2.5-coder:32b (this will take ~30 minutes for ~18GB download)..."
    ollama pull qwen2.5-coder:32b
fi

# Download base model for training
echo ""
echo "[7/7] Verifying Qwen2.5-Coder-32B-Instruct from Hugging Face..."
echo "(Model will be downloaded during first training run)"
python3 << 'EOF'
from transformers import AutoTokenizer
import os

cache_dir = os.path.expanduser("~/.cache/huggingface")
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

print(f"Verifying tokenizer access...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("✓ Tokenizer accessible")
except Exception as e:
    print(f"WARNING: Could not access tokenizer: {e}")
    print("Model will be downloaded during training")

print("✓ Model will be loaded in 4-bit with Unsloth during training")
EOF

# Create directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/{triplets,selected,checkpoints,logs,tensorboard}
mkdir -p data
echo "✓ Directories created"

# Initialize database
echo ""
echo "Initializing database schema..."
if [ -f "/home/user/mcp_servers/polymax-synthesizer/papers.db" ]; then
    sqlite3 /home/user/mcp_servers/polymax-synthesizer/papers.db < data/schema.sql
    echo "✓ Database schema initialized"
else
    echo "WARNING: papers.db not found. Will be created on first run."
fi

# Summary
echo ""
echo "======================================================"
echo "✓ Setup Complete! (v0.2.0)"
echo "======================================================"
echo ""
echo "CRITICAL FIXES APPLIED:"
echo "  ✓ Model standardized to Qwen2.5-Coder-32B"
echo "  ✓ Unsloth installed for VRAM optimization"
echo "  ✓ dirtyjson for robust JSON parsing"
echo "  ✓ Parallelization support added"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run pilot extraction: python scripts/phase1_extract_triplets.py --config configs/extraction_config.yaml --pilot-mode --num-papers 20 --export"
echo "  3. Review quality in outputs/triplets/"
echo "  4. Proceed with full extraction if quality ≥7/10"
echo ""
echo "Storage usage:"
du -sh venv ~/.cache/huggingface/hub 2>/dev/null || echo "  (run after downloads complete)"
echo ""
echo "Estimated extraction time:"
echo "  20 papers (pilot): ~30-45 minutes (2 parallel workers)"
echo "  830 papers (full): 10-12 hours (2 parallel workers)"
