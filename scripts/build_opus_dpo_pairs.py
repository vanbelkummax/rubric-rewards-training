#!/usr/bin/env python3
"""
Build Opus DPO pairs focused on:
1. Reducing hallucinations (winner clean, loser hallucinates)
2. Maximizing novel hypotheses and logic quality

Strategy:
- Type A: Hallucination contrast (clean vs hallucinating)
- Type B: Quality contrast (high logic+novelty vs low)
- Type C: Combined (clean+high quality vs hallucinating+low quality)
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

INPUT_PATH = Path("/home/user/rubric-rewards-training/outputs/bioreasoner_21/candidates_tribunal_scored_combined.jsonl")
OUTPUT_PATH = Path("/home/user/rubric-rewards-training/outputs/bioreasoner_21/opus_dpo_pairs.jsonl")
DATA_OUTPUT = Path("/home/user/rubric-rewards-training/data/dpo_v21_opus_pairs.jsonl")

def load_scored_candidates():
    """Load and group candidates by prompt_id"""
    with open(INPUT_PATH) as f:
        candidates = [json.loads(line) for line in f]

    # Group by prompt_id
    grouped = defaultdict(list)
    for c in candidates:
        grouped[c['prompt_id']].append(c)

    return grouped

def get_scores(candidate):
    """Extract scores from candidate (handle both score formats)"""
    # Try combined scores first
    if 'combined_scores' in candidate:
        scores = candidate['combined_scores']
    elif 'tribunal_scores' in candidate:
        scores = candidate['tribunal_scores']
    elif 'codex_scores' in candidate:
        scores = candidate['codex_scores']
    else:
        # Fallback to individual fields
        scores = {
            'hallucination': 5 if not candidate.get('hallucination', False) else 1,
            'logic': candidate.get('logic_score', 3),
            'novelty': candidate.get('novelty_score', 3)
        }

    return {
        'hallucination': scores.get('hallucination', 3),
        'logic': scores.get('logic', 3),
        'novelty': scores.get('novelty', 3),
        'has_think': candidate.get('has_think', False),
        'think_length': candidate.get('think_length', 0)
    }

def compute_quality_score(scores):
    """Compute overall quality score (logic + novelty)"""
    return scores['logic'] + scores['novelty']

def is_clean(scores):
    """Check if candidate has no hallucination"""
    return scores['hallucination'] >= 4  # 4 or 5 means clean

def build_type_a_pairs(grouped):
    """
    Type A: Hallucination contrast pairs
    Winner: clean (no hallucination)
    Loser: hallucinates
    """
    pairs = []

    for prompt_id, candidates in grouped.items():
        clean = []
        hallucinating = []

        for c in candidates:
            scores = get_scores(c)
            if not scores['has_think']:
                continue  # Skip if no think block

            if is_clean(scores):
                clean.append((c, scores))
            else:
                hallucinating.append((c, scores))

        # Create pairs: best clean vs worst hallucinating
        if clean and hallucinating:
            # Sort clean by quality (descending)
            clean.sort(key=lambda x: compute_quality_score(x[1]), reverse=True)
            # Sort hallucinating by quality (ascending) - worst first
            hallucinating.sort(key=lambda x: compute_quality_score(x[1]))

            winner, w_scores = clean[0]
            loser, l_scores = hallucinating[0]

            pairs.append({
                'prompt_id': prompt_id,
                'prompt': winner['prompt'],
                'prompt_type': winner.get('prompt_type', ''),
                'papers': winner.get('papers', []),
                'chosen': winner['response'],
                'rejected': loser['response'],
                'pair_type': 'hallucination_contrast',
                'winner_scores': w_scores,
                'loser_scores': l_scores,
                'delta': {
                    'hallucination': w_scores['hallucination'] - l_scores['hallucination'],
                    'quality': compute_quality_score(w_scores) - compute_quality_score(l_scores)
                }
            })

    return pairs

def build_type_b_pairs(grouped):
    """
    Type B: Quality contrast pairs (within clean candidates only)
    Winner: high logic + novelty
    Loser: low logic + novelty
    Both must be clean (no hallucination)
    """
    pairs = []
    MIN_QUALITY_DELTA = 2.0  # Require meaningful gap

    for prompt_id, candidates in grouped.items():
        clean = []

        for c in candidates:
            scores = get_scores(c)
            if not scores['has_think']:
                continue
            if is_clean(scores):
                clean.append((c, scores))

        if len(clean) >= 2:
            # Sort by quality
            clean.sort(key=lambda x: compute_quality_score(x[1]), reverse=True)

            winner, w_scores = clean[0]
            loser, l_scores = clean[-1]  # Lowest quality among clean

            quality_delta = compute_quality_score(w_scores) - compute_quality_score(l_scores)

            if quality_delta >= MIN_QUALITY_DELTA:
                pairs.append({
                    'prompt_id': prompt_id,
                    'prompt': winner['prompt'],
                    'prompt_type': winner.get('prompt_type', ''),
                    'papers': winner.get('papers', []),
                    'chosen': winner['response'],
                    'rejected': loser['response'],
                    'pair_type': 'quality_contrast',
                    'winner_scores': w_scores,
                    'loser_scores': l_scores,
                    'delta': {
                        'hallucination': 0,  # Both clean
                        'quality': quality_delta
                    }
                })

    return pairs

def build_type_c_pairs(grouped):
    """
    Type C: Combined contrast (maximum separation)
    Winner: clean + high quality
    Loser: hallucinating + low quality
    """
    pairs = []

    for prompt_id, candidates in grouped.items():
        best_clean = None
        best_clean_score = -1
        worst_halluc = None
        worst_halluc_score = float('inf')

        for c in candidates:
            scores = get_scores(c)
            if not scores['has_think']:
                continue

            quality = compute_quality_score(scores)

            if is_clean(scores) and quality > best_clean_score:
                best_clean = (c, scores)
                best_clean_score = quality

            if not is_clean(scores) and quality < worst_halluc_score:
                worst_halluc = (c, scores)
                worst_halluc_score = quality

        if best_clean and worst_halluc:
            winner, w_scores = best_clean
            loser, l_scores = worst_halluc

            # Only include if there's substantial separation
            total_delta = (w_scores['hallucination'] - l_scores['hallucination']) + \
                         (compute_quality_score(w_scores) - compute_quality_score(l_scores))

            if total_delta >= 4.0:  # Strong contrast
                pairs.append({
                    'prompt_id': prompt_id,
                    'prompt': winner['prompt'],
                    'prompt_type': winner.get('prompt_type', ''),
                    'papers': winner.get('papers', []),
                    'chosen': winner['response'],
                    'rejected': loser['response'],
                    'pair_type': 'combined_contrast',
                    'winner_scores': w_scores,
                    'loser_scores': l_scores,
                    'delta': {
                        'hallucination': w_scores['hallucination'] - l_scores['hallucination'],
                        'quality': compute_quality_score(w_scores) - compute_quality_score(l_scores)
                    }
                })

    return pairs

def add_synthetic_exemplars():
    """
    Hand-crafted exemplars for edge cases:
    - Perfect grounding with novel hypothesis
    - Strong cross-paper reasoning
    """
    exemplars = [
        {
            'prompt_id': 'synthetic_grounding_1',
            'prompt': 'Given papers on spatial transcriptomics methods, propose a novel validation approach.',
            'prompt_type': 'synthetic',
            'papers': [],
            'chosen': '''<think>
The key challenge in spatial transcriptomics is validating predictions at single-cell resolution. Current methods rely on bulk correlation metrics that mask cell-type-specific errors.

I propose a hierarchical validation framework:
1. Cell-type stratified metrics (not just global SSIM)
2. Spatial autocorrelation analysis per gene
3. Cross-platform concordance using orthogonal technologies (FISH, ISH)

This addresses the limitation that high global scores can hide systematic failures in rare cell types.
</think>

Based on the methodological considerations above, I hypothesize that validation protocols stratified by cell type abundance will reveal performance gaps invisible to aggregate metrics. Specifically, rare populations (<5% of cells) likely show 2-3x higher prediction error than abundant types. This could be tested by comparing Visium HD predictions against matched MERFISH data for the same tissue section.''',
            'rejected': '''<think>
Spatial transcriptomics needs better validation.
</think>

More research is needed to validate spatial transcriptomics predictions. Current methods work well but could be improved. Future studies should look at this problem more carefully.''',
            'pair_type': 'synthetic_exemplar',
            'winner_scores': {'hallucination': 5, 'logic': 5, 'novelty': 5, 'has_think': True, 'think_length': 500},
            'loser_scores': {'hallucination': 5, 'logic': 1, 'novelty': 1, 'has_think': True, 'think_length': 50},
            'delta': {'hallucination': 0, 'quality': 8}
        },
        {
            'prompt_id': 'synthetic_cross_paper_1',
            'prompt': 'Bridge insights from digital pathology and single-cell genomics.',
            'prompt_type': 'synthetic',
            'papers': [],
            'chosen': '''<think>
Digital pathology excels at morphological pattern recognition but lacks molecular grounding. Single-cell genomics provides molecular resolution but loses spatial context. The bridge is spatial transcriptomics, but current methods don't fully exploit pathology foundation models.

Key insight: Pathology encoders (Virchow2, GigaPath) learn morphological features that correlate with transcriptional programs. If we can learn this mapping explicitly, we could:
1. Use pathology features to impute gene expression in archival H&E
2. Use expression patterns to guide attention in pathology models
3. Create hybrid representations that capture both modalities

The technical challenge is learning robust cross-modal embeddings without paired training data for most tissue types.
</think>

I propose a self-supervised framework that learns pathology-transcriptome alignment through:
1. Pseudo-pairing: Match morphologically similar regions across Visium HD samples
2. Contrastive learning: Pull together patches with similar expression profiles
3. Cross-prediction: Train bidirectional encoders (H&E→expression, expression→morphology descriptors)

Testable prediction: Models trained this way will show >10% improvement in gene expression prediction for tissue types absent from training, demonstrating generalizable morphology-expression relationships.''',
            'rejected': '''<think>
Digital pathology and single-cell genomics are both important fields.
</think>

These two fields could be combined in future research. Digital pathology looks at images and single-cell genomics looks at genes. Combining them would be useful for cancer research. More studies are needed to explore this connection.''',
            'pair_type': 'synthetic_exemplar',
            'winner_scores': {'hallucination': 5, 'logic': 5, 'novelty': 5, 'has_think': True, 'think_length': 700},
            'loser_scores': {'hallucination': 5, 'logic': 1, 'novelty': 1, 'has_think': True, 'think_length': 80},
            'delta': {'hallucination': 0, 'quality': 8}
        },
        {
            'prompt_id': 'synthetic_halluc_contrast_1',
            'prompt': 'Analyze the relationship between colibactin and colorectal cancer.',
            'prompt_type': 'synthetic',
            'papers': [{'pmid': '30742832', 'title': 'Colibactin DNA damage signature'}],
            'chosen': '''<think>
Colibactin is a genotoxin produced by pks+ E. coli that alkylates DNA, leaving characteristic mutational signatures SBS88 and ID18. The causal chain is:
1. pks+ E. coli colonization (common in infants, 56-66%)
2. Antibiotic exposure → dysbiosis → E. coli bloom
3. Colibactin-induced DNA damage → characteristic mutations
4. If damage hits tumor suppressors (APC, TP53) → cancer initiation

Evidence from the paper (PMID:30742832) shows the DNA damage signature is enriched in EOCRC.
</think>

Based on PMID:30742832, I hypothesize that antibiotic exposure in early childhood creates windows of vulnerability where pks+ E. coli blooms cause DNA damage that accumulates over decades, contributing to the rising EOCRC incidence. This predicts:
1. EOCRC patients will show higher SBS88 burden than late-onset CRC
2. Antibiotic exposure history will correlate with pks+ colonization
3. Protective interventions (probiotics, narrow-spectrum antibiotics) could reduce risk''',
            'rejected': '''<think>
Colibactin causes cancer through DNA damage. Studies by Smith et al. (PMID:99999999) and Jones et al. (PMID:88888888) showed this conclusively.
</think>

According to multiple studies including PMID:99999999 and PMID:88888888, colibactin from E. coli causes colorectal cancer. The mechanism involves DNA damage and mutation. Treatment options include antibiotics and surgery.''',
            'pair_type': 'synthetic_exemplar',
            'winner_scores': {'hallucination': 5, 'logic': 5, 'novelty': 4, 'has_think': True, 'think_length': 600},
            'loser_scores': {'hallucination': 1, 'logic': 2, 'novelty': 2, 'has_think': True, 'think_length': 150},
            'delta': {'hallucination': 4, 'quality': 4}
        },
        {
            'prompt_id': 'synthetic_methodology_1',
            'prompt': 'Propose improvements to current H&E to spatial transcriptomics prediction methods.',
            'prompt_type': 'synthetic',
            'papers': [],
            'chosen': '''<think>
Current H&E→ST methods (Img2ST, Hist2ST, HisToGene) share common limitations:
1. MSE loss assumes Gaussian errors, but expression is count data (Poisson/NB)
2. Fixed patch size ignores multi-scale morphology
3. No uncertainty quantification
4. Train on spot-level data but evaluate at higher resolution

Key insight: The sparsity trap - models learn to predict mean expression because variance is high. Breaking this requires:
- Loss functions that reward capturing expression variance
- Architectures that model gene-gene correlations
- Evaluation metrics beyond per-gene correlation
</think>

I propose three concrete improvements:
1. **Poisson-NB loss**: Replace MSE with negative log-likelihood under count distributions. This properly weights low-expression genes and models overdispersion.

2. **Multi-scale attention**: Process patches at 2μm, 8μm, and 32μm simultaneously, letting the model learn which scale matters for each gene. Cell morphology matters at fine scale; tissue architecture at coarse.

3. **Spatial consistency loss**: Penalize predictions that violate known spatial autocorrelation patterns. If neighboring spots have similar expression, predictions should too.

Testable: Models with these changes should show >15% improvement on held-out genes and tissue types, with calibrated uncertainty estimates.''',
            'rejected': '''<think>
H&E prediction methods could be improved with better models.
</think>

Current methods for predicting spatial transcriptomics from H&E images could be improved by using larger models and more data. Deep learning approaches show promise but need more research. Transfer learning from pathology models might help.''',
            'pair_type': 'synthetic_exemplar',
            'winner_scores': {'hallucination': 5, 'logic': 5, 'novelty': 5, 'has_think': True, 'think_length': 550},
            'loser_scores': {'hallucination': 5, 'logic': 2, 'novelty': 1, 'has_think': True, 'think_length': 100},
            'delta': {'hallucination': 0, 'quality': 7}
        }
    ]

    return exemplars

def main():
    print("Loading scored candidates...")
    grouped = load_scored_candidates()
    print(f"Loaded {sum(len(v) for v in grouped.values())} candidates across {len(grouped)} prompts")

    # Build all pair types
    print("\nBuilding Type A pairs (hallucination contrast)...")
    type_a = build_type_a_pairs(grouped)
    print(f"  Created {len(type_a)} pairs")

    print("Building Type B pairs (quality contrast)...")
    type_b = build_type_b_pairs(grouped)
    print(f"  Created {len(type_b)} pairs")

    print("Building Type C pairs (combined contrast)...")
    type_c = build_type_c_pairs(grouped)
    print(f"  Created {len(type_c)} pairs")

    print("Adding synthetic exemplars...")
    synthetic = add_synthetic_exemplars()
    print(f"  Added {len(synthetic)} exemplars")

    # Combine all pairs, avoiding duplicates
    all_pairs = []
    seen_prompts = set()

    # Priority: Type C > Type A > Type B (combined has strongest signal)
    for pair in type_c:
        if pair['prompt_id'] not in seen_prompts:
            all_pairs.append(pair)
            seen_prompts.add(pair['prompt_id'])

    for pair in type_a:
        if pair['prompt_id'] not in seen_prompts:
            all_pairs.append(pair)
            seen_prompts.add(pair['prompt_id'])

    for pair in type_b:
        if pair['prompt_id'] not in seen_prompts:
            all_pairs.append(pair)
            seen_prompts.add(pair['prompt_id'])

    # Always add synthetic
    all_pairs.extend(synthetic)

    # Sort by total delta (strongest pairs first)
    all_pairs.sort(
        key=lambda x: x['delta']['hallucination'] + x['delta']['quality'],
        reverse=True
    )

    # Save
    print(f"\nSaving {len(all_pairs)} total pairs...")

    for path in [OUTPUT_PATH, DATA_OUTPUT]:
        path.parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + '\n')
        print(f"  Saved to {path}")

    # Print summary
    print("\n" + "="*60)
    print("OPUS DPO PAIRS SUMMARY")
    print("="*60)
    print(f"Type A (hallucination contrast): {len(type_a)}")
    print(f"Type B (quality contrast): {len(type_b)}")
    print(f"Type C (combined contrast): {len(type_c)}")
    print(f"Synthetic exemplars: {len(synthetic)}")
    print(f"Total unique pairs: {len(all_pairs)}")

    # Breakdown by pair type in final set
    type_counts = defaultdict(int)
    for p in all_pairs:
        type_counts[p['pair_type']] += 1
    print("\nFinal composition:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Average deltas
    avg_halluc_delta = sum(p['delta']['hallucination'] for p in all_pairs) / len(all_pairs)
    avg_quality_delta = sum(p['delta']['quality'] for p in all_pairs) / len(all_pairs)
    print(f"\nAvg hallucination delta: {avg_halluc_delta:.2f}")
    print(f"Avg quality delta: {avg_quality_delta:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
