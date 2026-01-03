# BioReasoner v3.0: GRPO Implementation Plan

**Date**: 2026-01-03
**Status**: Planning Complete, Ready for Implementation
**Author**: Claude Opus 4.5 + Max Van Belkum

---

## Executive Summary

After v2.2 DPO failed to improve hallucination rates (42.0% vs 40.6%), we're pivoting from Direct Preference Optimization (DPO) to **Group Relative Policy Optimization (GRPO)** based on guidance from HuggingFace's "Smol Training Playbook."

### Why GRPO?

| Problem with DPO | GRPO Solution |
|------------------|---------------|
| Needed 100-500 pairs, we had 35 | No paired data required |
| Reward margin too weak (0.027) | Direct reward signal per sample |
| Manual pair curation exhausting | Automated scoring via reward function |
| Optimizes vague "preference" | Optimizes explicit hallucination metric |

---

## Phase 1: GRPO Reward Function

**Output**: `scripts/grpo_reward.py`

### Reward Design

```python
def compute_reward(response: str, valid_pmids: set) -> float:
    """
    Multi-objective reward function for BioReasoner v3.0

    Components:
    1. Citation Grounding (primary) - punish hallucinations harshly
    2. Logic Quality (secondary) - reward coherent reasoning
    3. Novelty (secondary) - reward non-generic hypotheses
    4. Format Compliance - require <think> blocks
    """
    reward = 0.0

    # === CITATION GROUNDING (weight: highest) ===
    cited_pmids = extract_pmids(response)
    n_valid = 0
    n_hallucinated = 0

    for pmid in cited_pmids:
        if pmid in valid_pmids:
            n_valid += 1
            reward += 1.0
        else:
            n_hallucinated += 1
            reward -= 2.0  # 2x penalty for hallucination

    # Bonus for citing without hallucinating
    if n_valid > 0 and n_hallucinated == 0:
        reward += 1.0  # Clean citation bonus

    # === LOGIC QUALITY ===
    logic_score = score_logic(response)  # 1-5 scale
    reward += (logic_score - 1) / 4.0  # Normalize to 0-1

    # === NOVELTY ===
    novelty_score = score_novelty(response)  # 1-5 scale
    reward += (novelty_score - 1) / 4.0  # Normalize to 0-1

    # === FORMAT COMPLIANCE ===
    if has_think_block(response):
        reward += 0.5
    else:
        reward -= 1.0  # Penalty for missing <think>

    return reward
```

### Expected Reward Distribution

| Response Type | Typical Reward |
|---------------|----------------|
| Perfect (clean cites, high quality) | +4.0 to +6.0 |
| Good (clean cites, medium quality) | +2.0 to +4.0 |
| Mediocre (no cites, medium quality) | +0.5 to +1.5 |
| Bad (1 hallucination) | -1.0 to +0.5 |
| Terrible (multiple hallucinations) | -4.0 to -2.0 |

---

## Phase 2: Prompt Dataset Preparation

**Output**: `data/grpo_prompts.jsonl`

### Source: Polymath Corpus

We have 15,294 papers harvested across domains:
- Category Theory, Cognitive Science, Network Science
- Biosemiotics, Cybernetics, Thermodynamics
- TRIZ, Operations Research, Spatial Statistics

### Prompt Generation Strategy

1. **Extract paper triplets** from polymath corpus
2. **Generate prompts** using same templates as v2.0:
   - `critique_extend`: Critique paper A, extend with insights from B and C
   - `method_bridge`: Bridge methodology from A to domain of B using C
   - `gap_analysis`: Identify gaps in A, propose solutions from B and C

3. **Include valid PMIDs** in prompt metadata for reward function

### Target Dataset Size

| Phase | Prompts | Purpose |
|-------|---------|---------|
| Initial | 500 | Quick iteration cycle |
| Full | 2,000 | Production training |
| Extended | 5,000+ | If more compute available |

---

## Phase 3: TRL GRPOTrainer Configuration

**Output**: `scripts/train_grpo.py`

### Training Configuration

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    # Model
    model_name_or_path="models/bioreasoner-2.1-merged",

    # GRPO-specific
    num_generations=4,  # Generate 4 responses per prompt
    temperature=0.7,

    # LoRA
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Training
    learning_rate=1e-6,  # Lower than DPO
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # Memory optimization
    bf16=True,
    gradient_checkpointing=True,

    # Output
    output_dir="models/bioreasoner-3.0-grpo",
    logging_steps=10,
    save_steps=100,
)
```

### Key Differences from DPO

| Aspect | DPO | GRPO |
|--------|-----|------|
| Data | Paired (winner, loser) | Unpaired prompts |
| Reward | Implicit from pairs | Explicit function |
| Generations | 1 per prompt | N per prompt (e.g., 4) |
| Learning | Contrastive | Group-relative |

---

## Phase 4: Training Execution

### Hardware Requirements

- **GPU**: RTX 5090 24GB VRAM
- **Expected VRAM**: ~20GB with 4-bit quantization
- **Training time**: ~2-4 hours for 500 prompts

### Execution Steps

```bash
# 1. Prepare environment
cd /home/user/rubric-rewards-training
source venv/bin/activate

# 2. Generate prompt dataset
python scripts/prepare_grpo_prompts.py \
    --n_prompts 500 \
    --output data/grpo_prompts.jsonl

# 3. Run GRPO training
python scripts/train_grpo.py \
    --prompts data/grpo_prompts.jsonl \
    --output models/bioreasoner-3.0-grpo

# 4. Merge adapter
python scripts/merge_adapter.py \
    --base models/bioreasoner-2.1-merged \
    --adapter models/bioreasoner-3.0-grpo \
    --output models/bioreasoner-3.0-merged
```

### Monitoring

- Watch `rewards/mean` - should increase over training
- Watch `rewards/std` - should decrease (model becoming more consistent)
- Log hallucination rate per checkpoint

---

## Phase 5: Benchmark and Validation

### Tribunal Evaluation

Run on same held-out test set (83 samples) as v2.1/v2.2:

```bash
python scripts/tribunal_eval_v21.py \
    --model models/bioreasoner-3.0-merged \
    --output outputs/tribunal_v30
```

### Success Criteria

| Metric | v2.1 Baseline | v2.2 (Failed) | v3.0 Target |
|--------|---------------|---------------|-------------|
| Pass Rate | 59.4% | 58.0% | **>65%** |
| Hallucination Rate | 40.6% | 42.0% | **<25%** |
| Missing `<think>` | 0% | 0% | 0% |
| Avg Logic | 4.00 | 3.99 | >4.0 |
| Avg Novelty | 4.39 | 4.39 | >4.5 |

---

## Risk Mitigation

### Potential Issues

1. **Reward hacking**: Model learns to not cite at all
   - Mitigation: Add penalty for zero citations

2. **VRAM overflow**: 4 generations per prompt is memory-intensive
   - Mitigation: Use gradient checkpointing, reduce batch size

3. **Training instability**: GRPO can be sensitive to hyperparameters
   - Mitigation: Start with conservative LR (1e-6), monitor loss closely

4. **Reward function bugs**: Incorrect scoring leads to wrong optimization
   - Mitigation: Test reward function on known good/bad examples first

### Fallback Plan

If GRPO fails:
1. Return to DPO with 200+ manually curated pairs
2. Or: Fresh SFT from base with polymath corpus

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Implement reward function | `scripts/grpo_reward.py` |
| 1 | Prepare prompt dataset | `data/grpo_prompts.jsonl` |
| 2 | Configure and test trainer | `scripts/train_grpo.py` |
| 2-3 | Run GRPO training | `models/bioreasoner-3.0-grpo/` |
| 3 | Merge and benchmark | Final results |

---

## References

1. [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) - HuggingFace, Oct 2025
2. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
3. [TRL GRPOTrainer Documentation](https://huggingface.co/docs/trl/grpo_trainer)
4. [Dr. GRPO: Understanding and Fixing Biases in GRPO](https://arxiv.org/abs/2503.XXXXX)

---

## Appendix: File Inventory

### Scripts to Create

| File | Purpose |
|------|---------|
| `scripts/grpo_reward.py` | Reward function implementation |
| `scripts/prepare_grpo_prompts.py` | Generate prompts from polymath corpus |
| `scripts/train_grpo.py` | Main GRPO training script |

### Existing Scripts to Reuse

| File | Purpose |
|------|---------|
| `scripts/merge_adapter.py` | Merge LoRA into base model |
| `scripts/tribunal_eval_v21.py` | Held-out test evaluation |
| `scripts/novelty_scoring.py` | Novelty scoring functions |

### Data Files

| File | Content |
|------|---------|
| `data/grpo_prompts.jsonl` | Prompts for GRPO training |
| `data/test_84.jsonl` | Held-out test set |
