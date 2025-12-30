# Project Status - Rubric Rewards Training

**Date**: 2025-12-30
**Status**: ✅ **AUDIT PHASE COMPLETE - READY FOR PILOT EXTRACTION**
**GitHub**: https://github.com/vanbelkummax/rubric-rewards-training

---

## ✅ Completed Components

### Phase 1: Extraction (READY)
- ✅ `scripts/phase1_extract_triplets.py` - Full extraction script with error handling
- ✅ `scripts/utils/prompts.py` - Extraction and grading prompts
- ✅ `configs/extraction_config.yaml` - Configuration with quality thresholds
- ✅ Database schema with research_triplets table
- ✅ JSON export functionality for review
- ✅ Quality statistics computation
- ✅ Retry logic and timeout handling
- ✅ Pilot mode support (20 papers)

**Next Action**: Run pilot extraction on 20 papers

```bash
cd /home/user/rubric-rewards-training
source venv/bin/activate  # After running setup.sh
python scripts/phase1_extract_triplets.py \
    --config configs/extraction_config.yaml \
    --pilot-mode \
    --num-papers 20 \
    --export
```

### Infrastructure (COMPLETE)
- ✅ Repository structure
- ✅ Comprehensive README with examples
- ✅ Database schema (SQLite)
- ✅ Configuration files (YAML)
- ✅ Setup script with dependency installation
- ✅ Example triplet with quality notes
- ✅ Documentation (DESIGN.md, AUDIT.md)
- ✅ MIT License
- ✅ .gitignore for Python/ML projects
- ✅ requirements.txt with all dependencies
- ✅ GitHub repository created and pushed

### Documentation (COMPLETE)
- ✅ Full audit report (AUDIT.md)
- ✅ Technical design (DESIGN.md)
- ✅ README with detailed instructions
- ✅ Example outputs
- ✅ Setup instructions
- ✅ Success criteria defined
- ✅ Abort conditions specified

---

## ⏳ Pending Components (Phases 2-3)

### Phase 2: Selection (NOT YET IMPLEMENTED)
- ⏳ `scripts/phase2_select_best.py` - Selection with self-critique
- ⏳ `configs/selection_config.yaml` - Selection configuration
- ⏳ Scoring logic implementation
- ⏳ Quality report generation

**Priority**: Implement after Phase 1 pilot succeeds

### Phase 3: Training (NOT YET IMPLEMENTED)
- ⏳ `scripts/phase3_train_grpo.py` - GRPO training implementation
- ⏳ `scripts/utils/grading.py` - Rubric grading with frozen model
- ⏳ `scripts/utils/monitoring.py` - VRAM and metric tracking
- ⏳ QLoRA setup code
- ⏳ GRPO algorithm implementation
- ⏳ Tensorboard logging
- ⏳ Checkpoint management

**Priority**: Implement after Phase 2 succeeds

### Testing (NOT YET IMPLEMENTED)
- ⏳ `tests/test_extraction.py`
- ⏳ `tests/test_selection.py`
- ⏳ `tests/test_grading.py`
- ⏳ `tests/test_integration.py`

**Priority**: Add as each phase is implemented

---

## 📊 Resource Verification

| Resource | Available | Required | Status |
|----------|-----------|----------|--------|
| Papers | 830 | 100+ | ✅ 8× more than needed |
| VRAM | 24GB | 20GB | ✅ 20% headroom |
| RAM | 196GB | 16GB | ✅ 12× more than needed |
| Storage | 188GB | 2.5GB | ✅ 75× more than needed |
| GPU Model | RTX 5090 | RTX 4090+ | ✅ Latest generation |

---

## 🎯 Implementation Roadmap

### Week 1: Pilot & Extraction (NOW)
- [x] Day 1: Setup environment, verify dependencies
- [ ] Day 2: Run pilot extraction (20 papers)
- [ ] Day 3: Manual quality review
- [ ] Day 4: Full extraction (830 papers) if quality ≥7/10
- [ ] Day 5: Export and analyze extraction stats

### Week 2: Selection (AFTER WEEK 1)
- [ ] Implement phase2_select_best.py
- [ ] Test on pilot data
- [ ] Run full selection
- [ ] Generate quality report
- [ ] CEO review of selected samples

### Week 3: Training Stage 1 (AFTER WEEK 2)
- [ ] Implement GRPO training script
- [ ] Set up QLoRA configuration
- [ ] Test on mini-batch (10 samples)
- [ ] Run Stage 1 training (830 samples, ~5 hours)
- [ ] Evaluate on validation set

### Week 4: Training Stage 2 & Integration (AFTER WEEK 3)
- [ ] Run Stage 2 fine-tuning (830 best, ~3.5 hours)
- [ ] Merge adapters
- [ ] Export to Ollama
- [ ] Add MCP tool to research-lab
- [ ] Final evaluation and writeup

---

## 🚨 Decision Points

### After Pilot Extraction (20 papers)
**Quality Check**: Manual review of 20 extracted triplets

✅ **GO if**:
- Research goals are specific and actionable (≥14/20 samples)
- Rubric items are measurable (≥14/20)
- No hallucinations detected
- Reference solutions match paper content

❌ **NO-GO if**:
- Quality <7/10 average
- Hallucination rate >15%
- Rubric items too vague or unmeasurable

**Fallback**: Tune extraction prompts, add few-shot examples, or reduce to manual curation

### After Full Extraction (830 papers)
**Quality Check**: Statistical analysis + random sampling

✅ **GO if**:
- Success rate >80% (≥664 papers)
- Rubric item distribution 5-8 items/sample
- Goal length distribution 50-200 words

❌ **NO-GO if**:
- Success rate <80%
- Quality degradation vs pilot

**Fallback**: Re-extract failed papers with adjusted prompts

### After Stage 1 Training
**Performance Check**: Validation set evaluation

✅ **GO if**:
- Rubric satisfaction >60% (stage 1 baseline)
- Training stability (smooth loss curve)
- VRAM usage <23GB

❌ **NO-GO if**:
- No improvement over base model
- Catastrophic forgetting (code capability drop >10%)
- VRAM overflow

**Fallback**: Adjust hyperparameters, reduce batch size, or use qwen3:14b

---

## 📈 Success Metrics

### Extraction (Phase 1)
- [ ] ≥664/830 papers successfully extracted (80%)
- [ ] Avg rubric items: 6 ± 1.5
- [ ] Avg goal length: 100 ± 50 words
- [ ] No hallucinations in random sample of 50

### Selection (Phase 2)
- [ ] 830 best triplets selected
- [ ] Avg selection score >7.0/10
- [ ] Domain coverage maintained (60% bio, 30% ML, 10% clinical)

### Training (Phase 3)
- [ ] Stage 1: Rubric satisfaction >60%
- [ ] Stage 2: Rubric satisfaction >70%
- [ ] Cross-domain performance drop <10%
- [ ] Base code capability retained >90%

---

## 🔗 Important Links

**GitHub Repository**: https://github.com/vanbelkummax/rubric-rewards-training

**Documentation**:
- [README](README.md) - Usage instructions
- [AUDIT.md](docs/AUDIT.md) - Full feasibility audit
- [DESIGN.md](docs/DESIGN.md) - Technical design document
- [SETUP](setup.sh) - Automated setup script

**Source Paper**:
- [arxiv 2512.23707](https://arxiv.org/abs/2512.23707) - Training AI Co-Scientists Using Rubric Rewards

**Local Files**:
- Full design: `/home/user/docs/plans/2025-12-30-rubric-rewards-local-training-design.md`
- Full audit: `/home/user/docs/RUBRIC_REWARDS_AUDIT.md`
- Database: `/home/user/mcp_servers/polymax-synthesizer/papers.db` (830 papers)

---

## 🎓 Learning Outcomes

**If Successful**:
- New research capability: Cross-domain research plan generation
- Methodology for fine-tuning local models with rubric rewards
- Dataset of 830 high-quality research triplets
- Template for future RL fine-tuning projects

**If Fails at Phase 1**:
- Insights into LLM extraction limitations
- Prompt engineering techniques
- Fallback to manual curation still valuable

**If Fails at Phase 3**:
- Understanding of GRPO algorithm challenges
- QLoRA optimization experience
- Dataset still usable for other training methods

---

**Status Summary**: ✅ Ready for pilot extraction. All infrastructure complete. Awaiting quality review before full extraction.

**Next Command**:
```bash
cd /home/user/rubric-rewards-training
bash setup.sh
# Then run pilot extraction as shown above
```
