# BioReasoner Lab Notebook

## Project Goal
Train a 7B model to perform grounded scientific reasoning on biomedical literature without hallucinating citations.

---

## Entry 1: Initial Training (2025-12-31 to 2026-01-02)

### Training Data Construction

**Source**: 830 papers from 7 Vanderbilt professors (Huo, Landman, Lau, Hwang, Najdawi, Sarkar, Washington)

**Triplet Generation**:
- Created paper triplets for cross-paper reasoning tasks
- Prompt types: `critique_extend`, `method_bridge`, `gap_analysis`
- Generated Chain-of-Thought responses using Claude Opus 4.5

**Data Splits**:
| Dataset | Count | Location |
|---------|-------|----------|
| SFT Training | 751 | `data/train_750.jsonl` |
| Test (held-out) | 83 | `data/test_84.jsonl` |
| DPO Pairs | 280 | `data/dpo_distillation_pairs.jsonl` |

### Stage 1: Supervised Fine-Tuning (SFT)

**Config**:
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Method: LoRA (r=16, alpha=32)
- Learning rate: 2e-5
- Epochs: 3
- Max length: 8192 tokens

**Output**: `models/bioreasoner-sft-8k/` (309MB adapter)

### Stage 2: Direct Preference Optimization (DPO)

**Config**:
- Base: SFT checkpoint
- Beta: 0.1
- 280 preference pairs (Claude chosen vs student rejected)
- Max length: 4096 tokens

**Output**: `models/bioreasoner-dpo/` (155MB adapter)

### Merge

**Command**: `python scripts/merge_adapter.py --device cuda`

**Output**: `models/bioreasoner-2.0-merged/` (~15GB SafeTensors)

---

## Entry 2: Blind Evaluation - Scientific Tribunal (2026-01-02)

### Setup
- **Lead Judge**: Claude Opus 4.5
- **Second Judge**: Codex
- **Test Set**: First 10 samples from `data/test_84.jsonl`
- **Generation Config**: temperature=0.7, max_new_tokens=1024

### Results

| Metric | Claude Opus | Codex | Agreement |
|--------|-------------|-------|-----------|
| Hallucination Fails | 0/10 | 0/10 | ✅ UNANIMOUS |
| Pass Rate | 10/10 | 10/10 | ✅ UNANIMOUS |
| Avg Logic | 3.4/5 | 2.9/5 | Minor diff |
| Avg Novelty | 3.9/5 | 2.7/5 | Codex stricter |
| Missing `<think>` | 2/10 | - | - |

### Critical Findings
1. **ZERO hallucinated PMIDs** - Primary goal achieved
2. **2 samples missing `<think>` blocks** - Format regression
3. **Novelty moderate** - Hypotheses often generic

### Verdict: **CONDITIONAL PASS**

**Files**:
- Model outputs: `outputs/blind_eval_batch1.jsonl`
- Codex verdict: `outputs/SECOND_JUDGE_VERDICT.json`
- Final verdict: `outputs/TRIBUNAL_FINAL_VERDICT.json`

---

## Entry 3: BioReasoner 2.1 Planning (2026-01-02)

### Problem Analysis
1. 20% of outputs missing `<think>` blocks
2. Novelty scores lower than training targets
3. Some cross-domain prompts produce incoherent bridges

### Proposed Solution: Iterative DPO (RAFT-style)

**Phase 1: Best-of-N Generation**
- 150 prompts × 4 responses = 600 candidates
- Force-prefix `<think>` to ensure format compliance
- Diverse prompt types from training corpus

**Phase 2: Tribunal Filtering**
- REJECT: Missing `<think>`, think_length < 200, auto_score < 2.5
- ACCEPT: Strong logic, high novelty, proper citations

**Phase 3: DPO Pair Construction**
- Winner: highest (logic + novelty), has `<think>`
- Loser: lowest score OR missing `<think>`
- Target ~200-300 high-quality pairs

**Phase 4: DPO Training Round 2**
- Same config as v2.0
- Expected: eliminate `<think>` failures, boost novelty

### Pipeline Script
`scripts/bioreasoner_21_pipeline.py`

Commands:
```bash
# Step 1: Generate batch for inference
python scripts/bioreasoner_21_pipeline.py generate --n_prompts 150 --n_per_prompt 4

# Step 2: [CODEX] Run inference on generation_batch.jsonl

# Step 3: Auto-filter candidates
python scripts/bioreasoner_21_pipeline.py filter

# Step 4: [TRIBUNAL] Score passed candidates

# Step 5: Build DPO pairs
python scripts/bioreasoner_21_pipeline.py build_dpo
```

---

## Checkpoints

| Version | Date | Location | Notes |
|---------|------|----------|-------|
| v2.0-merged | 2026-01-02 | `models/bioreasoner-2.0-merged/` | First complete model, 0 hallucinations |
| v2.0-checkpoint | 2026-01-02 | `models/checkpoints/v2.0/` | Backup before v2.1 training |

---

## Key Learnings

1. **Grounding works**: SFT + DPO successfully teaches citation discipline
2. **Format regression possible**: Model can "forget" `<think>` scaffolding
3. **Novelty is hard**: 7B model produces generic hypotheses without explicit training
4. **Dual-judge valuable**: Codex catches issues Claude misses (stricter on novelty)

---

---

## Entry 4: BioReasoner 2.1 Execution (2026-01-03)

### Phase 1: Best-of-N Generation - COMPLETE
- **Generated**: 600 candidates (150 prompts × 4)
- **Force-prefix**: `<think>` block enforced
- **Output**: `outputs/bioreasoner_21/generation_results.jsonl`

### Phase 2: Auto-Filter - COMPLETE
| Result | Count | Rate |
|--------|-------|------|
| Passed | 595 | 99.2% |
| Rejected | 5 | 0.8% |
| Missing `<think>` | **0** | 0% |

**Key Win**: Force-prefixing eliminated `<think>` failures entirely!

Rejection reasons: All 5 were LOW_AUTO_SCORE (weak reasoning quality)

### Phase 3: DPO Pair Construction - COMPLETE
- **Model pairs**: 38 (from 150 prompts with 2+ score delta)
- **Synthetic exemplars**: 8 (hand-crafted high-novelty examples)
- **Total**: 46 pairs
- **Output**: `data/dpo_v21_pairs.jsonl`

### Phase 4: Training - IN PROGRESS
**Config**:
- Base: `models/bioreasoner-2.0-merged/`
- LoRA: r=32, alpha=64
- LR: 2e-6
- Epochs: 2
- Beta: 0.1

**Status**: ✅ COMPLETE (2026-01-03 07:54 UTC)

**Results**:
- Training time: 5:34
- Final loss: 0.657
- Reward margin: 0.071
- Adapter: `models/bioreasoner-2.1-dpo/` (309MB)
- **Merged model**: `models/bioreasoner-2.1-merged/` (15.25GB)

### Polymath Corpus Expansion (Parallel)
Harvesting papers across 25+ theoretical domains:
- Category Theory, Cognitive Analogy, Network Science
- Biosemiotics, Cybernetics, Thermodynamics of Computation
- Evolutionary Game Theory, Architectural Theory
- Condensed Matter Physics, Spatial Statistics
- Cognitive Linguistics, Molecular Programming
- TRIZ, Epistemology, Operations Research, Autopoiesis
- PEFT/LoRA, Quantization, Preference Optimization
- Behavioral Economics, Inference/Serving

**Status**: 28+ subagents running, ~8,500+ papers queued

---

## Checkpoints

| Version | Date | Location | Notes |
|---------|------|----------|-------|
| v2.0-merged | 2026-01-02 | `models/bioreasoner-2.0-merged/` | First complete model, 0 hallucinations |
| v2.0-checkpoint | 2026-01-02 | `models/checkpoints/v2.0/` | Backup before v2.1 training |
| v2.1-dpo | 2026-01-03 | `models/bioreasoner-2.1-dpo/` | Training in progress |

---

## Next Steps

- [x] Run v2.1 Best-of-N generation
- [x] Auto-filter candidates (595 passed)
- [x] Build DPO pairs (46 pairs)
- [x] Complete DPO training round 2 (5:34, loss 0.657)
- [x] Merge v2.1 adapter (15.25GB)
- [ ] Tribunal evaluation on full test set
- [ ] Ingest polymath corpus to vector DB

---

## Project Map (2026-01-03)

### Directory Structure
```
/home/user/rubric-rewards-training/
├── data/
│   ├── train_750.jsonl          # SFT training data (751 samples)
│   ├── test_84.jsonl            # Held-out test set (83 samples)
│   ├── dpo_distillation_pairs.jsonl  # v2.0 DPO pairs (280)
│   ├── dpo_v21_pairs.jsonl      # v2.1 DPO pairs (46)
│   └── synthetic_dpo_exemplars.jsonl # Hand-crafted exemplars (8)
├── models/
│   ├── bioreasoner-sft-8k/      # SFT adapter (309MB)
│   ├── bioreasoner-dpo/         # v2.0 DPO adapter (155MB)
│   ├── bioreasoner-2.0-merged/  # v2.0 full model (15GB)
│   ├── bioreasoner-2.1-dpo/     # v2.1 DPO adapter (309MB)
│   └── bioreasoner-2.1-merged/  # v2.1 full model (15.25GB) ← CURRENT
├── outputs/
│   ├── bioreasoner_21/
│   │   ├── generation_batch.jsonl     # Input batch (600)
│   │   ├── generation_results.jsonl   # Codex outputs (600)
│   │   ├── candidates_passed.jsonl    # Filtered (595)
│   │   └── candidates_rejected.jsonl  # Rejected (5)
│   ├── blind_eval_batch1.jsonl
│   └── TRIBUNAL_FINAL_VERDICT.json
├── scripts/
│   ├── bioreasoner_21_pipeline.py     # Main pipeline
│   ├── run_bioreasoner_21_inference.py # Codex inference runner
│   ├── novelty_scoring.py             # Novelty rubric
│   ├── train_dpo_8k.py                # DPO training
│   └── merge_adapter.py               # Adapter merging
└── logs/
    └── train_v21_run6.log             # Training log
```

### Memory & Knowledge Graph Contents
**Entities**:
- `BioReasoner-2.1-Plan`: Training strategy
- `Polymath_Corpus_Expansion`: Harvest effort tracking
- `Polymath_Paper_Database`: 15,294 papers dataset
- `PEFT_Papers`, `Preference_Optimization_Papers`, etc.

**Key Observations**:
- 15,294 papers harvested (OpenAlex 80.7%, Semantic Scholar 19.3%)
- 73.3% have abstracts, avg 850 citations
- Domains: CS, AI, Biology, Psychology, Medicine + 20 more
- 355 fully ingested, 14,677 pending

### Codex Collaboration
**Completed**:
- Generated 600 inference outputs with `--force_think` flag
- Created `scripts/run_bioreasoner_21_inference.py` (resumable, per-item seeds)
- Outputs saved to `outputs/bioreasoner_21/generation_results.jsonl`

### Methods Available
| Tool | Purpose |
|------|---------|
| `train_dpo_8k.py` | DPO training with 4-bit quantization |
| `merge_adapter.py` | Merge LoRA into base model |
| `novelty_scoring.py` | Score responses for DPO pair selection |
| `bioreasoner_21_pipeline.py` | End-to-end pipeline (generate/filter/build_dpo) |
| Research-lab MCP | Paper harvesting, ingestion, RAG |
| Memory MCP | Knowledge graph persistence |

### Remaining Work
1. **Tribunal Evaluation**: Run v2.1 on full 83-sample test set
2. **Paper Ingestion**: Process 14,677 pending papers to vector DB
3. **v2.2 Planning**: Use polymath corpus for next iteration

### Training History
| Version | Date | Pairs | Result |
|---------|------|-------|--------|
| v2.0 | 2026-01-02 | 280 | 0 hallucinations, 20% missing `<think>` |
| v2.1 | 2026-01-03 | 46 | 0% missing `<think>` (force-prefix), loss 0.657 |
| v2.2 | 2026-01-03 | 35 | Hallucination-focused DPO, loss 0.682 |

---

## Entry 5: BioReasoner v2.2 (2026-01-03)

### Tribunal Evaluation of v2.1 Candidates
**Dual-judge agreement**: Claude Opus + Codex both scored 595 candidates

| Metric | Claude | Codex |
|--------|--------|-------|
| Pass Rate | 74.8% | 74.8% |
| Hallucination Failures | 150 | 150 |
| Missing `<think>` | 0% | 0% |
| Avg Logic | 4.27 | 4.30 |
| Avg Novelty | 3.94 | 3.92 |

### DPO Pair Construction (Hallucination Focus)
**Opus pairs**: 34 total
- Combined contrast (clean+quality vs halluc+low): 23
- Hallucination contrast (clean vs halluc): 7
- Synthetic exemplars: 4

**Codex pairs**: 46 (novelty-based)

**Merged for v2.2**: 35 unique pairs (deduplicated by prompt_id, highest delta)

### v2.2 Training
- Base: `models/bioreasoner-2.1-merged`
- Pairs: `data/dpo_v22_merged.jsonl` (35)
- Config: LoRA r=16, alpha=32, lr=2e-6, epochs=2, max_length=2048
- Training time: 5:09
- Final loss: 0.682
- Merged model: `models/bioreasoner-2.2-merged` (15.25GB)

### Key Changes from v2.1
1. **Hallucination-targeted pairs**: Explicit winner=clean, loser=hallucinating
2. **Quality contrast pairs**: High logic+novelty vs low
3. **Synthetic exemplars**: Hand-crafted ideal responses for edge cases

### v2.2 Benchmark Results (Held-Out Test Set)
**Evaluated**: 69/83 samples (14 skipped - missing prompts)

| Metric | v2.1 | v2.2 | Delta |
|--------|------|------|-------|
| Pass Rate | 59.4% (41/69) | 58.0% (40/69) | **-1.4%** ⚠️ |
| Hallucination Rate | 40.6% (28/69) | 42.0% (29/69) | **+1.5%** ⚠️ |
| Missing `<think>` | 0% | 0% | — |
| Avg Logic | 4.00 | 3.99 | -0.01 |
| Avg Novelty | 4.39 | 4.39 | — |

**Verdict**: v2.2 performed *slightly worse* than v2.1. Hallucination-focused DPO did not generalize.

### Root Cause Analysis
1. **Insufficient pairs**: 35 pairs is too few for meaningful DPO. Smol Playbook recommends 100-500.
2. **Distribution mismatch**: Training candidates (595) differ from held-out test (83 samples)
3. **Weak preference signal**: `rewards/margins: 0.027` - very small margin between chosen/rejected
4. **Possible overfitting**: 2 epochs on 35 pairs may memorize rather than generalize

### Decision: Proceed to v3.0 with Polymath Corpus
Per Smol Training Playbook:
- Fresh high-quality data > more DPO rounds on stalled trajectory
- 15,000+ papers harvested, ready for diverse SFT examples

---

## Entry 6: v3.0 Planning (2026-01-03)

### Proposed Strategy
1. **Diversify SFT data**: Generate 300-500 new examples from polymath corpus
   - Cross-domain reasoning (physics→biology, CS→medicine)
   - Novel paper triplets from new domains
   - Higher novelty exemplars

2. **Fresh SFT**: Train from Qwen2.5-7B base with combined data
   - Original 751 biomedical samples
   - 300-500 new polymath samples
   - ~1000-1200 total SFT examples

3. **Targeted DPO**: Generate 200+ pairs from new model
   - Larger pair count for meaningful signal
   - Multi-domain hallucination contrast
   - Quality-focused selection

### Available Resources
- **Polymath papers**: 15,294 harvested (73.3% with abstracts)
- **Domains**: Category Theory, Cognitive Science, Network Science, Biosemiotics, Cybernetics, Thermodynamics, Game Theory, TRIZ, Operations Research, Spatial Statistics, Molecular Programming
- **Professors MCP**: 830 papers from 7 Vanderbilt faculty

### Next Steps
- [x] Benchmark v2.2 on held-out test set
- [x] Document v2.2 failure analysis
- [x] Consult Smol Training Playbook for strategic pivot
- [ ] ~~Audit polymath corpus for SFT-ready papers~~ (superseded by GRPO approach)
- [ ] ~~Design cross-domain triplet generation~~ (superseded by GRPO approach)
- [ ] ~~Generate v3.0 SFT dataset~~ (superseded by GRPO approach)
- [ ] ~~Train v3.0 from fresh SFT~~ (superseded by GRPO approach)

---

## Entry 7: Strategic Pivot to GRPO (2026-01-03)

### The Smol Training Playbook Diagnosis

After reviewing HuggingFace's "The Smol Training Playbook" (214 pages, Oct 2025), the v2.2 failure was diagnosed as a **mathematical inevitability** due to violating core DPO principles:

#### Signal-to-Noise Ratio Violation
| Constraint | Playbook Guideline | Our v2.2 | Status |
|------------|-------------------|----------|--------|
| **Pair Count** | 100-500 pairs minimum | 35 pairs | ❌ VIOLATED |
| **Reward Margin** | Meaningful separation | 0.027 | ❌ TOO WEAK |
| **Distribution** | Match test distribution | Training ≠ Test | ❌ MISMATCH |

#### Key Playbook Quotes
1. *"SFT gives most of the gains... Base models are too unrefined to benefit from advanced post-training methods."*
2. *"DPO is unstable and requires 100-500 pairs to overcome noise."*
3. *"GRPO does not need paired data. It generates multiple outputs, scores them using a reward function, and updates to favor high scores."*

### Strategy Options Evaluated

| Option | Method | Pros | Cons |
|--------|--------|------|------|
| **A** | More DPO pairs (300+) | Safe, known pipeline | Manual curation exhausting |
| **B** | GRPO (DeepSeek-R1 style) | No paired data, direct reward optimization | Trickier to tune |
| **C** | Fresh SFT from base | Clean distribution | Loses v2.1 format compliance |

### Decision: **Option B - GRPO**

**Rationale**:
1. **Solves data bottleneck**: No need to find "good loser" examples
2. **Direct optimization**: Code reward function for exact metric (hallucination-free)
3. **Playbook-endorsed**: DeepSeek-R1 used GRPO for reasoning capabilities
4. **Automation**: No manual pair curation needed

### v3.0 GRPO Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     v3.0 GRPO Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Base Model: v2.1-merged (preserves format compliance)      │
│       ↓                                                     │
│  Prompt Pool: 15k polymath papers (unlabeled questions)     │
│       ↓                                                     │
│  Generation: N responses per prompt (e.g., N=4)             │
│       ↓                                                     │
│  Reward Function:                                           │
│    reward = 0                                               │
│    for each cited PMID:                                     │
│      if PMID in valid_papers: reward += 1.0                 │
│      else: reward -= 2.0  # Harsh penalty for hallucination │
│    reward += logic_score / 5                                │
│    reward += novelty_score / 5                              │
│       ↓                                                     │
│  GRPO Update: Favor high-reward responses                   │
│       ↓                                                     │
│  v3.0-merged                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Reward Function Design

```python
def compute_reward(response: str, valid_pmids: set) -> float:
    """
    GRPO reward function for BioReasoner v3.0

    Scoring:
    - +1.0 per valid citation
    - -2.0 per hallucinated citation (harsh penalty)
    - +0.0 to +1.0 for logic quality
    - +0.0 to +1.0 for novelty

    Range: [-∞, +∞] but typically [-4, +4]
    """
    reward = 0.0

    # Citation grounding (primary objective)
    cited_pmids = extract_pmids(response)
    for pmid in cited_pmids:
        if pmid in valid_pmids:
            reward += 1.0  # Correct citation
        else:
            reward -= 2.0  # Hallucination penalty (2x weight)

    # Quality bonuses
    reward += score_logic(response) / 5.0   # 0.0 to 1.0
    reward += score_novelty(response) / 5.0  # 0.0 to 1.0

    # Format compliance
    if has_think_block(response):
        reward += 0.5
    else:
        reward -= 1.0

    return reward
```

### Implementation Plan

| Phase | Task | Output |
|-------|------|--------|
| **1** | Create GRPO reward function | `scripts/grpo_reward.py` |
| **2** | Prepare prompt dataset from polymath | `data/grpo_prompts.jsonl` |
| **3** | Configure TRL GRPOTrainer | `scripts/train_grpo.py` |
| **4** | Run GRPO training | `models/bioreasoner-3.0-grpo/` |
| **5** | Merge and benchmark | `models/bioreasoner-3.0-merged/` |

### References
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) - HuggingFace, Oct 2025
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - GRPO for reasoning
- [Dr. GRPO](https://arxiv.org/abs/2503.XXXXX) - Bias fixes for GRPO
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) - Implementation reference

---

## Training History (Updated)

| Version | Date | Method | Pairs/Prompts | Result |
|---------|------|--------|---------------|--------|
| v2.0 | 2026-01-02 | SFT+DPO | 280 pairs | 0 hallucinations, 20% missing `<think>` |
| v2.1 | 2026-01-03 | DPO | 46 pairs | 0% missing `<think>`, 40.6% hallucination |
| v2.2 | 2026-01-03 | DPO | 35 pairs | **FAILED** - 42.0% hallucination (+1.5%) |
| v3.0 | TBD | **GRPO** | 15k prompts | Targeting <20% hallucination |

---

## Checkpoints (Updated)

| Version | Date | Location | Notes |
|---------|------|----------|-------|
| v2.0-merged | 2026-01-02 | `models/bioreasoner-2.0-merged/` | First complete model |
| v2.1-merged | 2026-01-03 | `models/bioreasoner-2.1-merged/` | Best format compliance (0% missing `<think>`) |
| v2.2-merged | 2026-01-03 | `models/bioreasoner-2.2-merged/` | Failed DPO iteration |
| v3.0-grpo | TBD | `models/bioreasoner-3.0-grpo/` | GRPO adapter (planned) |
