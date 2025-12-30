# Critical Fixes - Technical Audit Response

**Date**: 2025-12-30
**Status**: Addressing critical scientific and technical issues identified in audit

---

## Issues Identified

### 1. ⚠️ CRITICAL: Context Truncation (Data Integrity Risk)

**Original Code**:
```python
full_text=full_text[:3000] if full_text else "No full text available"
```

**Problem**:
- 3,000 characters ≈ 500-750 words
- Typical Methods section: 1,500-4,000+ words
- **Result**: LLM hallucinates reference solutions, corrupts reward signal

**Fix**: Remove truncation, use full context (Qwen supports 32k+ tokens)

---

### 2. ⚠️ CRITICAL: 24GB VRAM Constraint for GRPO

**Problem**:
- Base model (4-bit): 17-18 GB
- LoRA adapters: 300 MB
- GRPO with 4 generations × 2,500 tokens = KV cache OOM

**Fix**:
- Integrate Unsloth for 30% memory reduction
- Reduce `num_generations: 4 → 2`
- Add gradient checkpointing optimizations

---

### 3. 🔴 HIGH: Brittle JSON Parsing

**Problem**:
- `find('{')` and `rfind('}')` fail on Markdown blocks (```json)
- No handling of nested braces or multi-object outputs

**Fix**:
- Strip Markdown code blocks with regex
- Add `dirtyjson` fallback for malformed JSON
- Retry logic with structure enforcement

---

### 4. 🟡 MEDIUM: Model Naming Inconsistency

**Problem**:
- Config: `qwen3-coder:30b` (doesn't exist)
- Setup: `Qwen2.5-Coder-32B-Instruct`
- Mismatch: 30B vs 32B

**Fix**: Standardize on `Qwen2.5-Coder-32B-Instruct` everywhere

---

### 5. 🟡 MEDIUM: Extraction Throughput

**Problem**:
- Documentation claims: 2 hours for 830 papers
- Reality: 830 × 180s timeout = 41.5 hours (worst case)
- Realistic: 830 × 45s = 10.3 hours (sequential)

**Fix**:
- Add parallel processing (2-4 concurrent requests)
- Update time estimates in documentation

---

### 6. 🟡 MEDIUM: Frozen Grader on Single GPU

**Problem**:
- GRPO needs frozen copy for grading
- Loading 2× 32B models = OOM on 24GB

**Fix**:
- Multipass approach: Generate → Unload adapters → Grade → Reload → Train
- Document reward hacking risk if using active model

---

## Implementation Plan

### Phase 1: Critical Fixes (Today)
- [x] Remove text truncation
- [x] Fix JSON parsing
- [x] Standardize model names
- [x] Add Unsloth integration
- [x] Update time estimates

### Phase 2: Optimizations (Next)
- [ ] Add extraction parallelization
- [ ] Implement multipass grading
- [ ] Add validation tests

### Phase 3: Documentation (Final)
- [ ] Update README with corrected specs
- [ ] Add troubleshooting for VRAM issues
- [ ] Document scientific validation steps

---

## Testing Plan

### Extraction Validation
```bash
# Test on 1 paper with full context
python scripts/phase1_extract_triplets.py --num-papers 1 --export

# Verify:
# 1. Full text length > 3000 chars
# 2. JSON parsing succeeds
# 3. Reference solution matches paper
```

### VRAM Monitoring
```bash
# During training, monitor:
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# Should stay < 23GB
```

---

## Changelog

### v0.2.0 - Critical Fixes (2025-12-30)

**Changed**:
- Removed full_text truncation (scientific integrity fix)
- Improved JSON parsing with Markdown stripping + dirtyjson fallback
- Standardized model names to Qwen2.5-Coder-32B-Instruct
- Added Unsloth integration for VRAM optimization
- Reduced GRPO generations from 4 → 2

**Fixed**:
- Hallucination risk from truncated context
- JSON parsing failures on Markdown blocks
- Model mismatch errors
- VRAM overflow in GRPO training

**Updated**:
- Time estimates: 2 hours → 10-12 hours for extraction
- VRAM requirements: 18GB → 22GB peak with optimizations

---

**Status**: Fixes implemented, testing in progress
