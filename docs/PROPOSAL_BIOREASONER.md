# Project Title

**BioReasoner: Distilling Rigorous Scientific Reasoning into Consumer-Deployable Language Models via Iterative Tribunal Alignment**

---

## Research Proposal

### The Problem: Hallucination Kills Trust

Large language models excel at scientific prose but fail at scientific *rigor*. In biomedical research, a single fabricated citation or mechanistically implausible hypothesis destroys credibility. Current solutions—retrieval-augmented generation, fine-tuning on PubMed—reduce but do not eliminate hallucinations. We address this **grounding-novelty tradeoff**: models that never hallucinate tend to be boring; models that generate novel hypotheses tend to fabricate support.

Our goal is a **7B-parameter model that runs on a single consumer GPU (RTX 3080/4080)** while achieving zero hallucinations and generating genuinely novel, testable hypotheses grounded in real literature.

### Methodology: Iterative Tribunal Alignment

We develop **BioReasoner** through a three-stage Constitutional Alignment pipeline:

**Stage 1: Supervised Fine-Tuning on Curated Triplets**

Starting from Qwen2.5-7B-Instruct, we fine-tune on 750 expert-curated paper triplets from Vanderbilt faculty publications spanning computational pathology, spatial transcriptomics, and clinical AI. Each training example requires cross-paper synthesis with explicit chain-of-thought reasoning in `<think>` blocks.

**Stage 2: Dual-Judge Tribunal Evaluation**

We employ a **Scientific Tribunal** where two frontier models (Claude Opus and a second judge) independently score model outputs on:
- **Hallucination** (5=all citations valid, 1=fabricated PMIDs)
- **Logic** (depth of cross-paper reasoning)
- **Novelty** (specificity and testability of hypotheses)

Consensus verdicts identify high-quality winners and low-quality losers for preference learning.

**Stage 3: Iterative DPO with Hallucination Targeting**

We construct DPO pairs prioritizing:
1. **Hallucination contrast**: Clean responses vs. hallucinating responses
2. **Quality contrast**: High logic+novelty vs. generic responses
3. **Synthetic exemplars**: Hand-crafted ideal responses for edge cases

Each iteration (v2.0 → v2.1 → v2.2) targets specific failure modes identified by the Tribunal, creating a self-improving loop.

### Polymathic Expansion

Beyond biomedical papers, we harvest 15,000+ papers from 25 theoretical domains—category theory, cybernetics, TRIZ methodology, thermodynamics of computation, evolutionary game theory—to teach cross-domain analogical reasoning. The model learns to bridge concepts (e.g., treating tumor evolution as a preference optimization problem, or cell-cell communication as message passing in graph neural networks).

### Current Results

| Version | Hallucination Rate | Missing Format | Avg Logic | Avg Novelty |
|---------|-------------------|----------------|-----------|-------------|
| v2.0 | 0% | 20% | 3.4 | 3.9 |
| v2.1 | 0% | **0%** | 4.3 | 3.9 |
| v2.2 | In training | — | — | — |

Key achievement: **Zero hallucinated citations** across all evaluated samples, with forced `<think>` scaffolding ensuring transparent reasoning.

### Deliverables

1. **BioReasoner-7B**: Open-weights model deployable on consumer GPUs (12GB+ VRAM)
2. **Tribunal Framework**: Reusable dual-judge evaluation pipeline
3. **Polymath Corpus**: 15K+ cross-domain papers for reasoning augmentation
4. **Training Recipes**: Reproducible SFT→DPO pipeline with all hyperparameters

### Timeline

- **Weeks 1-2**: SFT training + v2.0 DPO (complete)
- **Weeks 3-4**: Tribunal evaluation + v2.1/v2.2 iterations (in progress)
- **Weeks 5-6**: Polymath corpus ingestion + v3.0 training
- **Weeks 7-8**: Held-out benchmarking + model release

### Impact

A rigorously aligned 7B model running on consumer hardware democratizes AI-assisted hypothesis generation for academic labs without cloud API budgets. By open-sourcing the Tribunal framework, we enable other domains to replicate this Constitutional Alignment approach for their own scientific disciplines.

---

*Word count: 498*
