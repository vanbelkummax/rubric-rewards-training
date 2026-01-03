#!/usr/bin/env python3
"""
Merge Opus + Codex DPO pairs for v2.2 training.
Deduplicate by prompt_id, prefer higher delta pairs.
"""

import json
from pathlib import Path
from collections import defaultdict

# Input files
OPUS_PAIRS = Path("/home/user/rubric-rewards-training/data/dpo_v21_opus_pairs.jsonl")
CODEX_PAIRS = Path("/home/user/rubric-rewards-training/data/dpo_v21_pairs.jsonl")
CODEX_COMBINED = Path("/home/user/rubric-rewards-training/data/dpo_v21_pairs_combined.jsonl")

# Output
OUTPUT_PATH = Path("/home/user/rubric-rewards-training/data/dpo_v22_merged.jsonl")

def load_pairs(path):
    """Load pairs from JSONL file"""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_delta_score(pair):
    """Calculate total delta score for ranking"""
    if 'delta' in pair:
        return pair['delta'].get('hallucination', 0) + pair['delta'].get('quality', 0)
    # Fallback for different formats
    if 'winner_scores' in pair and 'loser_scores' in pair:
        w = pair['winner_scores']
        l = pair['loser_scores']
        return (w.get('hallucination', 0) - l.get('hallucination', 0)) + \
               (w.get('logic', 0) + w.get('novelty', 0) - l.get('logic', 0) - l.get('novelty', 0))
    return 0

def normalize_pair(pair, source):
    """Normalize pair format and add source tag"""
    normalized = {
        'prompt_id': pair.get('prompt_id', pair.get('group_id', 'unknown')),
        'prompt': pair.get('prompt', ''),
        'prompt_type': pair.get('prompt_type', ''),
        'papers': pair.get('papers', []),
        'chosen': pair.get('chosen', pair.get('winner', '')),
        'rejected': pair.get('rejected', pair.get('loser', '')),
        'source': source,
        'pair_type': pair.get('pair_type', 'unknown'),
        'delta_score': get_delta_score(pair)
    }

    # Preserve score details if available
    if 'winner_scores' in pair:
        normalized['winner_scores'] = pair['winner_scores']
    if 'loser_scores' in pair:
        normalized['loser_scores'] = pair['loser_scores']
    if 'delta' in pair:
        normalized['delta'] = pair['delta']

    return normalized

def main():
    print("Loading Opus pairs...")
    opus = load_pairs(OPUS_PAIRS)
    print(f"  Loaded {len(opus)} Opus pairs")

    print("Loading Codex pairs...")
    codex = load_pairs(CODEX_PAIRS)
    print(f"  Loaded {len(codex)} Codex pairs (novelty-based)")

    codex_combined = load_pairs(CODEX_COMBINED)
    print(f"  Loaded {len(codex_combined)} Codex pairs (tribunal-based)")

    # Normalize all pairs
    all_pairs = []
    for p in opus:
        all_pairs.append(normalize_pair(p, 'opus'))
    for p in codex:
        all_pairs.append(normalize_pair(p, 'codex_novelty'))
    for p in codex_combined:
        all_pairs.append(normalize_pair(p, 'codex_tribunal'))

    print(f"\nTotal pairs before dedup: {len(all_pairs)}")

    # Deduplicate by prompt_id, keeping highest delta
    by_prompt = defaultdict(list)
    for p in all_pairs:
        by_prompt[p['prompt_id']].append(p)

    merged = []
    for prompt_id, pairs in by_prompt.items():
        # Sort by delta score descending
        pairs.sort(key=lambda x: x['delta_score'], reverse=True)
        # Keep the best one
        best = pairs[0]
        # Tag if there were multiple sources
        best['alt_sources'] = [p['source'] for p in pairs[1:]] if len(pairs) > 1 else []
        merged.append(best)

    # Sort final set by delta score
    merged.sort(key=lambda x: x['delta_score'], reverse=True)

    print(f"After dedup: {len(merged)} unique pairs")

    # Save
    with open(OUTPUT_PATH, 'w') as f:
        for p in merged:
            f.write(json.dumps(p) + '\n')
    print(f"\nSaved to {OUTPUT_PATH}")

    # Summary statistics
    print("\n" + "="*60)
    print("MERGED DPO PAIRS SUMMARY (v2.2)")
    print("="*60)

    source_counts = defaultdict(int)
    type_counts = defaultdict(int)
    for p in merged:
        source_counts[p['source']] += 1
        type_counts[p['pair_type']] += 1

    print("\nBy source (primary):")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")

    print("\nBy pair type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    avg_delta = sum(p['delta_score'] for p in merged) / len(merged)
    print(f"\nAvg delta score: {avg_delta:.2f}")
    print(f"Max delta: {merged[0]['delta_score']:.2f}")
    print(f"Min delta: {merged[-1]['delta_score']:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
