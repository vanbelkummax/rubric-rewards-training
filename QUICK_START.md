# Quick Start Guide

**⚡ Get running in 3 commands**

## 1. Setup (2.5 hours - mostly downloads)

```bash
cd /home/user/rubric-rewards-training
bash setup.sh
```

This will:
- Install Python dependencies
- Download qwen3-coder:30b model (~30GB)
- Download Qwen2.5-Coder-32B tokenizer
- Create output directories
- Initialize database schema

## 2. Pilot Extraction (30 minutes)

```bash
source venv/bin/activate
python scripts/phase1_extract_triplets.py \
    --config configs/extraction_config.yaml \
    --pilot-mode \
    --num-papers 20 \
    --export
```

**Output**: `outputs/triplets/all_triplets.json`

## 3. Review Quality

```bash
cat outputs/triplets/all_triplets.json | head -100
```

**Quality Checklist**:
- [ ] Research goals are specific (not vague like "improve methods")
- [ ] Rubric items are measurable (have concrete thresholds)
- [ ] No hallucinations (all details match paper content)
- [ ] Reference solutions are accurate summaries

**Decision**:
- ✅ Quality ≥7/10 → Proceed to full extraction
- ❌ Quality <7/10 → Review `outputs/logs/extraction.log`, tune prompts

---

## Full Extraction (if pilot succeeds)

```bash
python scripts/phase1_extract_triplets.py \
    --config configs/extraction_config.yaml \
    --num-papers 830 \
    --export

# Expected: 2,490 triplets in 2 hours
# Check: outputs/triplets/extraction_stats.json
```

---

## Troubleshooting

**Ollama connection error**:
```bash
ollama serve  # In separate terminal
```

**Model not found**:
```bash
ollama pull qwen3-coder:30b
```

**Database not found**:
```bash
# Check polymax database exists
ls -lh /home/user/mcp_servers/polymax-synthesizer/papers.db
```

**Out of memory**:
```bash
# Reduce batch processing in config
# Or close other applications
```

---

## Next Steps

After successful full extraction:
1. Implement Phase 2 selection script
2. Run selection to pick best 830 triplets
3. Implement Phase 3 GRPO training
4. Train Stage 1 (5 hours)
5. Train Stage 2 (3.5 hours)
6. Integrate with research-lab MCP

---

**Questions?** See [README.md](README.md) or [PROJECT_STATUS.md](PROJECT_STATUS.md)
