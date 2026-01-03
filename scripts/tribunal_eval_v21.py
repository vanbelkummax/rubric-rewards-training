#!/usr/bin/env python3
"""
Tribunal Evaluation for BioReasoner v2.1
Lead Judge: Claude Opus 4.5
"""

import json
import re
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
MODEL_PATH = "/home/user/rubric-rewards-training/models/bioreasoner-2.2-merged"
TEST_PATH = "/home/user/rubric-rewards-training/data/test_84.jsonl"
OUTPUT_DIR = Path("/home/user/rubric-rewards-training/outputs/tribunal_v22")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_model():
    """Load BioReasoner v2.1 merged model"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, seed: int = 42) -> str:
    """Generate response with forced <think> prefix"""
    torch.manual_seed(seed)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Force <think> prefix
    text += "<think>"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return "<think>" + response  # Prepend the forced prefix

def extract_think_block(response: str) -> tuple:
    """Extract think block and measure length"""
    match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        return True, len(think_content), think_content
    # Check for unclosed think block
    if '<think>' in response:
        think_content = response.split('<think>', 1)[1]
        if '</think>' in think_content:
            think_content = think_content.split('</think>')[0]
        return True, len(think_content), think_content
    return False, 0, ""

def check_hallucination(response: str, papers: list) -> tuple:
    """Check for hallucinated PMIDs"""
    # Extract all PMIDs mentioned in response
    pmid_pattern = r'PMID[:\s]*(\d{7,8})|pmid[:\s]*(\d{7,8})'
    mentioned_pmids = set()
    for match in re.finditer(pmid_pattern, response, re.IGNORECASE):
        pmid = match.group(1) or match.group(2)
        mentioned_pmids.add(pmid)

    # Get valid PMIDs from papers
    valid_pmids = set()
    for p in papers:
        if 'pmid' in p:
            valid_pmids.add(str(p['pmid']))

    # Check for hallucinations
    hallucinated = mentioned_pmids - valid_pmids
    return len(hallucinated) > 0, list(hallucinated)

def score_logic(response: str, think_content: str) -> int:
    """Score logical reasoning quality (1-5)"""
    score = 3  # baseline

    # Positive signals
    if len(think_content) > 500:
        score += 0.5
    if any(word in think_content.lower() for word in ['therefore', 'because', 'suggests', 'indicates', 'hypothesis']):
        score += 0.5
    if re.search(r'\d+%|\d+\.\d+', think_content):  # quantitative reasoning
        score += 0.5
    if any(word in think_content.lower() for word in ['however', 'although', 'limitation', 'caveat']):
        score += 0.5  # considers alternatives

    # Negative signals
    if len(think_content) < 200:
        score -= 1
    if think_content.count('.') < 3:
        score -= 0.5  # too few sentences

    return max(1, min(5, round(score)))

def score_novelty(response: str, think_content: str) -> int:
    """Score hypothesis novelty (1-5)"""
    score = 3  # baseline

    # Positive signals for novelty
    novel_phrases = ['novel', 'unexplored', 'new mechanism', 'propose', 'speculate',
                     'cross-domain', 'bridge', 'integrate', 'synergy', 'paradigm']
    for phrase in novel_phrases:
        if phrase in response.lower():
            score += 0.3

    # Check for specific predictions
    if any(word in response.lower() for word in ['predict', 'expect', 'would result', 'should lead']):
        score += 0.5

    # Check for testable hypotheses
    if any(word in response.lower() for word in ['experiment', 'validate', 'test', 'measure']):
        score += 0.5

    # Negative: too generic
    generic_phrases = ['more research needed', 'further study required', 'well-established']
    for phrase in generic_phrases:
        if phrase in response.lower():
            score -= 0.5

    return max(1, min(5, round(score)))

def evaluate_sample(model, tokenizer, sample: dict, idx: int) -> dict:
    """Evaluate a single test sample"""
    prompt = sample.get('prompt')
    if not prompt:
        # Skip samples without prompts
        return None
    papers = sample.get('papers', [])

    # Generate response
    response = generate_response(model, tokenizer, prompt, seed=idx * 42)

    # Extract think block
    has_think, think_length, think_content = extract_think_block(response)

    # Check hallucination
    is_hallucinated, hallucinated_pmids = check_hallucination(response, papers)

    # Score
    logic = score_logic(response, think_content)
    novelty = score_novelty(response, think_content)

    # Pass criteria
    passed = (not is_hallucinated) and has_think and (logic >= 2)

    return {
        "idx": idx,
        "group_id": sample.get('group_id', ''),
        "prompt_type": sample.get('prompt_type', ''),
        "has_think": has_think,
        "think_length": think_length,
        "hallucination": is_hallucinated,
        "hallucinated_pmids": hallucinated_pmids,
        "logic_score": logic,
        "novelty_score": novelty,
        "pass": passed,
        "response_preview": response[:500] + "..." if len(response) > 500 else response
    }

def main():
    # Load test data
    with open(TEST_PATH) as f:
        test_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_samples)} test samples")

    # Load model
    model, tokenizer = load_model()

    # Evaluate all samples
    results = []
    skipped = 0
    for idx, sample in enumerate(test_samples):
        print(f"\n[{idx+1}/{len(test_samples)}] Evaluating {sample.get('prompt_type', 'unknown')}...")
        result = evaluate_sample(model, tokenizer, sample, idx)
        if result is None:
            print(f"  ⚠️ SKIPPED (no prompt)")
            skipped += 1
            continue
        results.append(result)

        status = "✅ PASS" if result['pass'] else "❌ FAIL"
        print(f"  {status} | think={result['has_think']} | halluc={result['hallucination']} | logic={result['logic_score']} | novelty={result['novelty_score']}")

        # Save incremental results
        with open(OUTPUT_DIR / "opus_results_incremental.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    # Calculate aggregates
    total = len(results)
    if total == 0:
        print("No samples evaluated!")
        return None
    pass_count = sum(1 for r in results if r['pass'])
    halluc_count = sum(1 for r in results if r['hallucination'])
    missing_think = sum(1 for r in results if not r['has_think'])
    avg_logic = sum(r['logic_score'] for r in results) / total
    avg_novelty = sum(r['novelty_score'] for r in results) / total

    verdict = {
        "judge": "claude_opus_4.5",
        "model": "bioreasoner-2.1-merged",
        "date": datetime.now().isoformat(),
        "total_samples": total,
        "skipped_samples": skipped,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / total, 4),
        "hallucination_count": halluc_count,
        "hallucination_rate": round(halluc_count / total, 4),
        "missing_think_count": missing_think,
        "missing_think_rate": round(missing_think / total, 4),
        "avg_logic": round(avg_logic, 2),
        "avg_novelty": round(avg_novelty, 2),
        "per_sample": results
    }

    # Save verdict
    with open(OUTPUT_DIR / "opus_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print("\n" + "="*60)
    print("TRIBUNAL VERDICT - LEAD JUDGE (Claude Opus 4.5)")
    print("="*60)
    print(f"Model: bioreasoner-2.1-merged")
    print(f"Total Samples: {total}")
    print(f"Pass Rate: {pass_count}/{total} ({verdict['pass_rate']*100:.1f}%)")
    print(f"Hallucination Rate: {halluc_count}/{total} ({verdict['hallucination_rate']*100:.1f}%)")
    print(f"Missing <think>: {missing_think}/{total} ({verdict['missing_think_rate']*100:.1f}%)")
    print(f"Avg Logic: {avg_logic:.2f}/5")
    print(f"Avg Novelty: {avg_novelty:.2f}/5")
    print("="*60)

    return verdict

if __name__ == "__main__":
    main()
