#!/usr/bin/env python3
"""
Tribunal Scoring for existing BioReasoner v2.1 generations.
Evaluates without regenerating - just scores the existing responses.
"""

import json
import re
from pathlib import Path
from datetime import datetime

INPUT_PATH = Path("/home/user/rubric-rewards-training/outputs/bioreasoner_21/candidates_passed.jsonl")
OUTPUT_PATH = Path("/home/user/rubric-rewards-training/outputs/bioreasoner_21/candidates_tribunal_scored.jsonl")

def extract_allowed_pmids(papers: list) -> set:
    """Extract PMIDs from papers list"""
    pmids = set()
    for p in papers:
        if isinstance(p, dict):
            if 'pmid' in p:
                pmids.add(str(p['pmid']))
        elif isinstance(p, str):
            pmids.add(p)
    return pmids

def extract_cited_pmids(response: str) -> set:
    """Extract PMIDs cited in response"""
    # Match PMID: 12345678, PMID#12345678, PMID 12345678, pmid:12345678
    # Also handle decimal IDs like 2512.20557 (arxiv-style)
    pattern = r'PMID[:#]?\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)'
    matches = re.findall(pattern, response, re.IGNORECASE)
    return set(matches)

def score_hallucination(response: str, papers: list) -> tuple:
    """Score hallucination: 5 if all cited PMIDs valid, 1 if any fabricated"""
    allowed = extract_allowed_pmids(papers)
    cited = extract_cited_pmids(response)

    if not cited:
        # No PMIDs cited - conservative pass (can't hallucinate what you don't cite)
        return 5, [], "no_pmids_cited"

    hallucinated = cited - allowed
    if hallucinated:
        return 1, list(hallucinated), "hallucinated_pmids"
    return 5, [], "all_pmids_valid"

def extract_think_block(response: str) -> tuple:
    """Extract think block info"""
    has_open = '<think>' in response
    has_close = '</think>' in response

    if has_open and has_close:
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return True, len(match.group(1).strip())
    elif has_open:
        # Unclosed think block
        content = response.split('<think>', 1)[1]
        return True, len(content.strip())

    return False, 0

def score_logic(response: str, think_content_len: int) -> int:
    """Score logical reasoning quality (1-5)"""
    score = 3.0  # baseline

    response_lower = response.lower()

    # Positive signals
    if think_content_len > 500:
        score += 0.5
    if think_content_len > 1000:
        score += 0.3

    reasoning_markers = ['therefore', 'because', 'suggests', 'indicates',
                        'hypothesis', 'consequently', 'thus', 'implies']
    for marker in reasoning_markers:
        if marker in response_lower:
            score += 0.15

    # Quantitative reasoning
    if re.search(r'\d+%|\d+\.\d+|p\s*[<>=]', response):
        score += 0.4

    # Considers alternatives/limitations
    critical_markers = ['however', 'although', 'limitation', 'caveat',
                       'alternatively', 'on the other hand']
    for marker in critical_markers:
        if marker in response_lower:
            score += 0.2

    # Cross-paper integration
    if 'paper' in response_lower and any(w in response_lower for w in ['both', 'together', 'combine', 'integrate']):
        score += 0.3

    # Negative signals
    if think_content_len < 200:
        score -= 1.0
    if think_content_len < 100:
        score -= 0.5

    return max(1, min(5, round(score)))

def score_novelty(response: str) -> int:
    """Score hypothesis novelty (1-5)"""
    score = 3.0  # baseline

    response_lower = response.lower()

    # Strong novelty signals
    novel_phrases = ['novel', 'unexplored', 'new mechanism', 'propose that',
                    'speculate', 'cross-domain', 'bridge', 'integrate',
                    'synergy', 'paradigm', 'unprecedented', 'first to']
    for phrase in novel_phrases:
        if phrase in response_lower:
            score += 0.25

    # Specific predictions
    prediction_markers = ['predict', 'expect', 'would result', 'should lead',
                         'likely to', 'anticipated', 'projected']
    for marker in prediction_markers:
        if marker in response_lower:
            score += 0.2

    # Testable/actionable hypotheses
    testable_markers = ['experiment', 'validate', 'test', 'measure',
                       'assay', 'protocol', 'crispr', 'knockout', 'knockdown']
    for marker in testable_markers:
        if marker in response_lower:
            score += 0.2

    # Specific methods mentioned
    if re.search(r'(RNA-seq|scRNA|spatial transcriptomics|ChIP-seq|ATAC-seq|mass spec)', response, re.IGNORECASE):
        score += 0.3

    # Negative: too generic
    generic_phrases = ['more research needed', 'further study required',
                      'well-established', 'it is known that', 'as expected']
    for phrase in generic_phrases:
        if phrase in response_lower:
            score -= 0.3

    return max(1, min(5, round(score)))

def main():
    print(f"Loading {INPUT_PATH}...")

    with open(INPUT_PATH) as f:
        samples = [json.loads(line) for line in f]

    print(f"Scoring {len(samples)} samples...")

    results = []
    stats = {
        'total': len(samples),
        'passed': 0,
        'failed_hallucination': 0,
        'failed_think': 0,
        'failed_logic': 0
    }

    for idx, sample in enumerate(samples):
        response = sample.get('response', '')
        papers = sample.get('papers', [])

        # Score hallucination
        halluc_score, halluc_pmids, halluc_note = score_hallucination(response, papers)

        # Think block
        has_think, think_length = extract_think_block(response)
        # Use existing values if available
        if 'has_think' in sample:
            has_think = sample['has_think']
        if 'think_length' in sample:
            think_length = sample['think_length']

        # Logic and novelty scores
        logic_score = score_logic(response, think_length)
        novelty_score = score_novelty(response)

        # Pass criteria
        passed = (halluc_score == 5) and has_think and (logic_score >= 2)

        # Update stats
        if passed:
            stats['passed'] += 1
        else:
            if halluc_score < 5:
                stats['failed_hallucination'] += 1
            if not has_think:
                stats['failed_think'] += 1
            if logic_score < 2:
                stats['failed_logic'] += 1

        # Build output
        scored_sample = {**sample}
        scored_sample['tribunal_scores'] = {
            'hallucination': halluc_score,
            'logic': logic_score,
            'novelty': novelty_score
        }
        scored_sample['has_think'] = has_think
        scored_sample['think_length'] = think_length
        scored_sample['pass'] = passed
        if halluc_pmids:
            scored_sample['notes'] = f"hallucinated_pmids: {halluc_pmids}"

        results.append(scored_sample)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(samples)}...")

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # Calculate aggregates
    avg_logic = sum(r['tribunal_scores']['logic'] for r in results) / len(results)
    avg_novelty = sum(r['tribunal_scores']['novelty'] for r in results) / len(results)
    avg_halluc = sum(r['tribunal_scores']['hallucination'] for r in results) / len(results)

    print("\n" + "="*60)
    print("TRIBUNAL SCORING COMPLETE")
    print("="*60)
    print(f"Total samples: {stats['total']}")
    print(f"Passed: {stats['passed']} ({100*stats['passed']/stats['total']:.1f}%)")
    print(f"Failed (hallucination): {stats['failed_hallucination']}")
    print(f"Failed (no <think>): {stats['failed_think']}")
    print(f"Failed (logic < 2): {stats['failed_logic']}")
    print(f"\nAvg Hallucination Score: {avg_halluc:.2f}/5")
    print(f"Avg Logic Score: {avg_logic:.2f}/5")
    print(f"Avg Novelty Score: {avg_novelty:.2f}/5")
    print(f"\nOutput: {OUTPUT_PATH}")
    print("="*60)

    # Save summary
    summary = {
        'date': datetime.now().isoformat(),
        'input_file': str(INPUT_PATH),
        'output_file': str(OUTPUT_PATH),
        'stats': stats,
        'avg_scores': {
            'hallucination': round(avg_halluc, 2),
            'logic': round(avg_logic, 2),
            'novelty': round(avg_novelty, 2)
        }
    }

    summary_path = OUTPUT_PATH.parent / "tribunal_scoring_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
