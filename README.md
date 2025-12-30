# Rubric Rewards Local Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Local implementation of Meta's "Rubric Rewards" methodology ([arxiv 2512.23707](https://arxiv.org/abs/2512.23707)) for training AI co-scientists using QLoRA on RTX 5090 (24GB VRAM).

**Author**: Max Van Belkum (Vanderbilt University)
**Date**: 2025-12-30
**Status**: Audit Phase - Implementation Ready

---

## 🎯 Project Overview

Train a local LLM (qwen3-coder:30b) to generate research plans using reinforcement learning with rubric-based self-grading. Applied to 830 papers across biomedical research, machine learning, and spatial transcriptomics.

### Key Adaptations from Paper
- ✅ **QLoRA** instead of full fine-tuning (24GB VRAM constraint)
- ✅ **Fully local** extraction pipeline (no API costs)
- ✅ **Staged training** for quality + diversity
- ✅ **830 papers** (paper used ~200-300)

### Expected Outcomes
- **Rubric satisfaction**: 70% (matches paper's expert approval rate)
- **Training time**: ~9 hours GPU + 2.5 hours extraction
- **Storage**: +2.5GB
- **New capability**: Cross-domain research plan generation

---

## 📊 Paper Corpus

**Source**: 830 papers from 7 Vanderbilt professors (polymax-synthesizer database)

| Professor | Papers | Domain |
|-----------|--------|--------|
| Bennett Landman | 280 | Medical imaging, brain MRI |
| Yuankai Huo | 211 | Computational pathology, Img2ST |
| Mary Kay Washington | 137 | GI pathology, IBD, cancer |
| Tae Hyun Hwang | 103 | Molecular AI, therapy prediction |
| Hirak Sarkar | 41 | Spatial transcriptomics methods |
| Ken Lau | 34 | CRC spatial atlas, single-cell |
| Fedaa Najdawi | 24 | Digital pathology, clinical AI |

**Coverage**: Biomedical (60%), ML (30%), Clinical AI (10%)

---

## 🖥️ Hardware Requirements

### Minimum
- **GPU**: 24GB VRAM (RTX 5090, A6000, RTX 4090)
- **RAM**: 16GB
- **Storage**: 50GB free (35GB dependencies + 15GB working)
- **CPU**: 8+ cores

### Tested On
```
GPU:  NVIDIA RTX 5090 Laptop (24GB VRAM)
RAM:  196GB
Disk: 188GB free
CPU:  24 cores
OS:   Ubuntu 22.04 (WSL2)
```

---

## 📁 Repository Structure

```
rubric-rewards-training/
├── scripts/
│   ├── phase1_extract_triplets.py      # Extract (goal, rubric, reference) from papers
│   ├── phase2_select_best.py           # Self-critique selection
│   ├── phase3_train_grpo.py            # GRPO training with QLoRA
│   ├── utils/
│   │   ├── database.py                 # SQLite database operations
│   │   ├── prompts.py                  # Extraction/selection prompts
│   │   ├── grading.py                  # Rubric grading with frozen model
│   │   └── monitoring.py               # Training monitoring & logging
├── configs/
│   ├── extraction_config.yaml          # Phase 1 settings
│   ├── selection_config.yaml           # Phase 2 settings
│   ├── training_config.yaml            # Phase 3 hyperparameters
│   └── qlora_config.yaml               # QLoRA configuration
├── data/
│   ├── schema.sql                      # Database schema
│   └── validation_split.json           # Hold-out papers for evaluation
├── outputs/
│   ├── triplets/                       # Extracted triplets (JSON)
│   ├── selected/                       # Best triplets per paper
│   ├── checkpoints/                    # Training checkpoints
│   └── logs/                           # Training logs
├── tests/
│   ├── test_extraction.py              # Unit tests for Phase 1
│   ├── test_selection.py               # Unit tests for Phase 2
│   ├── test_grading.py                 # Rubric grading validation
│   └── test_integration.py             # End-to-end tests
├── examples/
│   ├── sample_triplet.json             # Example training sample
│   ├── sample_plan.md                  # Example generated plan
│   └── pilot_results.md                # 20-paper pilot results
├── docs/
│   ├── DESIGN.md                       # Technical design document
│   ├── AUDIT.md                        # Full audit report
│   ├── API.md                          # MCP tool integration
│   └── TROUBLESHOOTING.md              # Common issues & solutions
├── requirements.txt                     # Python dependencies
├── setup.sh                            # One-command setup script
└── README.md                           # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/vanbelkummax/rubric-rewards-training.git
cd rubric-rewards-training

# Run setup (installs dependencies, downloads model)
bash setup.sh

# Estimated time: 2.5 hours (30GB model download)
```

### 2. Pilot Extraction (20 papers)
```bash
# Extract triplets from 20 papers for quality review
python scripts/phase1_extract_triplets.py \
    --config configs/extraction_config.yaml \
    --pilot-mode \
    --num-papers 20

# Review outputs in outputs/triplets/pilot/
# Expected: ~60 triplets (3 per paper)
```

### 3. Review Quality
```bash
# Manual review checklist (see examples/sample_triplet.json)
# ✅ Research goals are specific and actionable
# ✅ Rubrics have 5-8 measurable items
# ✅ Reference solutions match paper content
# ✅ No hallucinations or fabricated details

# If quality ≥7/10 → proceed to full extraction
# If quality <7/10 → tune prompts or use fallback
```

### 4. Full Extraction (830 papers)
```bash
# Extract all triplets (~2 hours)
python scripts/phase1_extract_triplets.py \
    --config configs/extraction_config.yaml \
    --num-papers 830

# Expected: ~2,490 triplets (3 per paper)
# Storage: ~10MB JSON
```

### 5. Select Best Triplets
```bash
# Self-critique selection (~30 mins)
python scripts/phase2_select_best.py \
    --config configs/selection_config.yaml

# Expected: 830 best triplets (1 per paper)
# Outputs quality metrics report
```

### 6. Train Model
```bash
# Stage 1: Train on all 2,490 samples (~5 hours)
python scripts/phase3_train_grpo.py \
    --config configs/training_config.yaml \
    --stage 1 \
    --samples all

# Stage 2: Fine-tune on 830 best (~3.5 hours)
python scripts/phase3_train_grpo.py \
    --config configs/training_config.yaml \
    --stage 2 \
    --samples best

# Monitor with: tensorboard --logdir outputs/logs
```

### 7. Evaluate & Export
```bash
# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/stage2_final \
    --validation data/validation_split.json

# Export to Ollama format
python scripts/export_to_ollama.py \
    --checkpoint outputs/checkpoints/stage2_final \
    --output qwen3-coder-rubric-rewards:latest
```

---

## 📋 Expected Outputs

### Phase 1: Extraction
```json
{
  "paper_id": 12345,
  "paper_title": "Img2ST: Imaging to Spatial Transcriptomics...",
  "professor": "Yuankai Huo",
  "samples": [
    {
      "research_goal": "Develop a method to predict spatial gene expression...",
      "rubric": [
        "Must handle Visium HD 2μm resolution",
        "Should outperform existing methods (SSIM >0.55)",
        "Must be computationally efficient (<10 min/sample)",
        "Should generalize across tissues",
        "Must preserve spatial structure"
      ],
      "reference_solution": "The paper proposes Img2ST-Net, a CNN-based..."
    }
  ]
}
```

### Phase 2: Selection
```
=== SELECTION QUALITY REPORT ===
Total triplets processed: 2,490
Best selected: 830

Rubric item distribution:
  5 items: 124 papers (15%)
  6 items: 298 papers (36%)
  7 items: 265 papers (32%)
  8 items: 143 papers (17%)

Goal length (words):
  Mean: 87, Median: 82, Range: 45-198

Domain coverage:
  Biomedical: 498 papers (60%)
  ML: 249 papers (30%)
  Clinical: 83 papers (10%)
```

### Phase 3: Training Metrics
```
=== STAGE 1 RESULTS ===
Epochs: 3
Steps: 156 (52 steps/epoch)
Time: 5.2 hours

Metrics (validation set):
  Rubric satisfaction: 68.4% (target: >70%)
  Plan length: 412 words avg
  Generation diversity: 0.83 unique bigrams
  Cross-domain transfer: -6.2% (biomedical → ML)

=== STAGE 2 RESULTS ===
Epochs: 2
Steps: 104
Time: 3.5 hours

Metrics (validation set):
  Rubric satisfaction: 73.1% ✅
  Plan length: 389 words avg
  Generation diversity: 0.87 unique bigrams
  Cross-domain transfer: -4.1% ✅
```

---

## 🔧 Configuration Files

### extraction_config.yaml
```yaml
# Phase 1: Triplet Extraction
model: "qwen3-coder:30b"
ollama_url: "http://localhost:11434/v1"

extraction:
  samples_per_paper: 3
  max_goal_words: 200
  rubric_items: [5, 8]  # Min, max
  max_reference_words: 500

database:
  path: "/home/user/mcp_servers/polymax-synthesizer/papers.db"
  output_table: "research_triplets"

prompts:
  extractor: "scripts/utils/prompts.py::EXTRACTOR_PROMPT"

timeout:
  per_paper: 180  # 3 minutes

retry:
  max_attempts: 3
  backoff: 5  # seconds
```

### training_config.yaml
```yaml
# Phase 3: GRPO Training
model:
  base: "Qwen/Qwen2.5-Coder-32B-Instruct"
  cache_dir: "/home/user/.cache/huggingface"

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  lora_r: 64
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  output_dir: "/home/user/work/rubric-rewards-training"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 100
  save_total_limit: 3

grpo:
  num_generations: 4  # N plans per goal
  frozen_grader: true
  length_penalty: 0.5
  max_plan_length: 500  # words
```

---

## 📊 Monitoring & Logging

### Real-Time Monitoring
```bash
# TensorBoard
tensorboard --logdir outputs/logs

# VRAM usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits'

# Training progress
tail -f outputs/logs/training.log
```

### Log Files
- `outputs/logs/extraction.log` - Phase 1 progress & errors
- `outputs/logs/selection.log` - Phase 2 scoring details
- `outputs/logs/training.log` - Phase 3 loss curves & metrics
- `outputs/logs/grading.log` - Rubric grading decisions

---

## ✅ Validation & Testing

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Test extraction only
pytest tests/test_extraction.py -v

# Test rubric grading
pytest tests/test_grading.py -v
```

### Integration Test
```bash
# End-to-end on 5 papers
python tests/test_integration.py --num-papers 5

# Expected: Extraction → Selection → Training on mini-batch
```

### Quality Checks
```bash
# Check for hallucinations in extracted triplets
python scripts/validate_triplets.py \
    --triplets outputs/triplets/all.json \
    --check-hallucinations

# Check rubric grading consistency
python scripts/validate_grading.py \
    --checkpoint outputs/checkpoints/stage1_final
```

---

## 🎯 Success Criteria

### Quantitative (Automated)
- ✅ Rubric satisfaction >70% on validation set
- ✅ Cross-domain performance drop <10%
- ✅ Training stability (smooth loss curve, no spikes)
- ✅ Plan length 300-500 words

### Qualitative (Manual Review)
- ✅ Plans are coherent and actionable
- ✅ Rubric compliance is genuine (not keyword matching)
- ✅ Model generalizes to new research goals
- ✅ Base code generation capability preserved (>90%)

---

## 🚨 Abort Conditions

### Phase 1 (Extraction)
- ❌ Quality <5/10 on 20-paper pilot → Manual curation fallback
- ❌ Extraction fails for >20% of papers → Investigate prompts
- ❌ Hallucination rate >15% → Add fact-checking step

### Phase 3 (Training)
- ❌ VRAM consistently >23GB → Use qwen3:14b (smaller model)
- ❌ No improvement over base after 2 epochs → Stop early
- ❌ Rubric grading too lenient (>95% satisfaction) → Adjust rubrics
- ❌ Base capability drops >10% → Lower learning rate

---

## 📚 Documentation

- **[DESIGN.md](docs/DESIGN.md)**: Full technical design (architecture, schemas, algorithms)
- **[AUDIT.md](docs/AUDIT.md)**: Feasibility audit with resource analysis
- **[API.md](docs/API.md)**: MCP tool integration for research-lab server
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues & solutions

---

## 🤝 Contributing

This is a research project for academic use. For issues or questions:
- **Email**: max.van.belkum@vanderbilt.edu
- **GitHub Issues**: Open an issue with detailed description
- **Pull Requests**: Welcome for bug fixes and improvements

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Meta Superintelligence Labs** for the Rubric Rewards paper ([arxiv 2512.23707](https://arxiv.org/abs/2512.23707))
- **Vanderbilt Professors** for the 830-paper corpus (Huo, Landman, Lau, Hwang, Najdawi, Sarkar, Washington)
- **QLoRA authors** for efficient fine-tuning method ([arxiv 2305.14314](https://arxiv.org/abs/2305.14314))
- **VeRL library** ([github.com/volcengine/verl](https://github.com/volcengine/verl)) for RL framework reference

---

## 📈 Roadmap

- [x] Design & audit (2025-12-30)
- [ ] Pilot extraction (20 papers)
- [ ] Full extraction (830 papers)
- [ ] Stage 1 training
- [ ] Stage 2 training
- [ ] MCP integration
- [ ] Publication writeup

---

**Last Updated**: 2025-12-30
**Version**: 0.1.0 (Audit Phase)
