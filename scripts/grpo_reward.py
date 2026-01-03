#!/usr/bin/env python3
"""
GRPO Reward Function for BioReasoner v3.0

This module provides the reward function used by TRL's GRPOTrainer
to optimize the model for grounded scientific reasoning.

Key objectives:
1. Eliminate hallucinated citations (primary)
2. Maintain logical reasoning quality
3. Encourage novel hypotheses
4. Enforce format compliance (<think> blocks)

Usage:
    from grpo_reward import BioReasonerRewardFunction

    reward_fn = BioReasonerRewardFunction()
    rewards = reward_fn(prompts, responses, valid_pmids_list)
"""

import re
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Configurable weights for reward components"""
    valid_citation: float = 1.0      # Reward per valid citation
    hallucination: float = -2.0      # Penalty per hallucinated citation
    clean_bonus: float = 1.0         # Bonus for citing without hallucinating
    no_citation_penalty: float = -0.5  # Penalty for not citing anything
    logic_weight: float = 1.0        # Weight for logic score (0-1 normalized)
    novelty_weight: float = 1.0      # Weight for novelty score (0-1 normalized)
    think_present: float = 0.5       # Bonus for having <think> block
    think_missing: float = -1.0      # Penalty for missing <think> block


class BioReasonerRewardFunction:
    """
    Reward function for GRPO training of BioReasoner.

    Computes a scalar reward for each (prompt, response) pair based on:
    - Citation grounding (hallucination detection)
    - Logical reasoning quality
    - Novelty of hypotheses
    - Format compliance
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()

        # PMID extraction pattern (handles various formats)
        self.pmid_pattern = r'PMID[:#]?\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)'

        # Logic indicators
        self.logic_positive = [
            'therefore', 'because', 'suggests', 'indicates', 'hypothesis',
            'consequently', 'thus', 'hence', 'implies', 'demonstrates'
        ]
        self.logic_alternatives = [
            'however', 'although', 'limitation', 'caveat', 'alternatively',
            'on the other hand', 'conversely', 'nonetheless'
        ]

        # Novelty indicators
        self.novelty_positive = [
            'novel', 'unexplored', 'new mechanism', 'propose', 'speculate',
            'cross-domain', 'bridge', 'integrate', 'synergy', 'paradigm',
            'unprecedented', 'first to', 'unique', 'innovative'
        ]
        self.novelty_predictions = [
            'predict', 'expect', 'would result', 'should lead',
            'anticipate', 'hypothesize', 'postulate'
        ]
        self.novelty_testable = [
            'experiment', 'validate', 'test', 'measure',
            'empirically', 'verify', 'demonstrate'
        ]
        self.novelty_generic = [
            'more research needed', 'further study required',
            'well-established', 'well-known', 'commonly accepted'
        ]

    def extract_pmids(self, text: str) -> Set[str]:
        """Extract all PMIDs mentioned in text"""
        matches = re.findall(self.pmid_pattern, text, re.IGNORECASE)
        return set(matches)

    def extract_think_block(self, response: str) -> Tuple[bool, str]:
        """Extract <think> block content"""
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return True, match.group(1).strip()

        # Check for unclosed think block
        if '<think>' in response:
            content = response.split('<think>', 1)[1]
            if '</think>' in content:
                content = content.split('</think>')[0]
            return True, content.strip()

        return False, ""

    def score_logic(self, response: str, think_content: str) -> float:
        """
        Score logical reasoning quality (0.0 to 1.0)

        Looks for:
        - Causal language (therefore, because, suggests)
        - Quantitative reasoning (percentages, numbers)
        - Consideration of alternatives/limitations
        - Sufficient depth (length, sentence count)
        """
        score = 0.5  # baseline

        # Positive: causal/logical language
        for word in self.logic_positive:
            if word in think_content.lower():
                score += 0.05
                break  # Only count once

        # Positive: considers alternatives
        for word in self.logic_alternatives:
            if word in think_content.lower():
                score += 0.1
                break

        # Positive: quantitative reasoning
        if re.search(r'\d+%|\d+\.\d+', think_content):
            score += 0.1

        # Positive: sufficient depth
        if len(think_content) > 500:
            score += 0.1
        elif len(think_content) > 300:
            score += 0.05

        # Positive: multiple sentences
        sentence_count = think_content.count('.')
        if sentence_count >= 5:
            score += 0.1
        elif sentence_count >= 3:
            score += 0.05

        # Negative: too short
        if len(think_content) < 100:
            score -= 0.3
        elif len(think_content) < 200:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def score_novelty(self, response: str, think_content: str) -> float:
        """
        Score hypothesis novelty (0.0 to 1.0)

        Looks for:
        - Novel/innovative language
        - Specific predictions
        - Testable hypotheses
        - Penalizes generic conclusions
        """
        full_text = (think_content + " " + response).lower()
        score = 0.5  # baseline

        # Positive: novel language
        novel_count = sum(1 for phrase in self.novelty_positive if phrase in full_text)
        score += min(0.2, novel_count * 0.05)

        # Positive: predictions
        for word in self.novelty_predictions:
            if word in full_text:
                score += 0.1
                break

        # Positive: testable
        for word in self.novelty_testable:
            if word in full_text:
                score += 0.1
                break

        # Negative: generic
        for phrase in self.novelty_generic:
            if phrase in full_text:
                score -= 0.15

        return max(0.0, min(1.0, score))

    def compute_reward(
        self,
        response: str,
        valid_pmids: Set[str]
    ) -> Tuple[float, dict]:
        """
        Compute reward for a single response.

        Args:
            response: Model-generated response text
            valid_pmids: Set of valid PMIDs from the prompt's papers

        Returns:
            Tuple of (reward_score, breakdown_dict)
        """
        w = self.weights
        breakdown = {}

        # === CITATION GROUNDING (Primary Objective) ===
        cited_pmids = self.extract_pmids(response)
        n_valid = len(cited_pmids & valid_pmids)
        n_hallucinated = len(cited_pmids - valid_pmids)
        n_total = len(cited_pmids)

        citation_reward = 0.0
        citation_reward += n_valid * w.valid_citation
        citation_reward += n_hallucinated * w.hallucination

        # Bonus for clean citations
        if n_valid > 0 and n_hallucinated == 0:
            citation_reward += w.clean_bonus

        # Penalty for not citing at all
        if n_total == 0:
            citation_reward += w.no_citation_penalty

        breakdown['citation_reward'] = citation_reward
        breakdown['n_valid_cites'] = n_valid
        breakdown['n_hallucinated'] = n_hallucinated

        # === THINK BLOCK ===
        has_think, think_content = self.extract_think_block(response)
        if has_think:
            format_reward = w.think_present
        else:
            format_reward = w.think_missing

        breakdown['has_think'] = has_think
        breakdown['format_reward'] = format_reward

        # === LOGIC QUALITY ===
        logic_score = self.score_logic(response, think_content)
        logic_reward = logic_score * w.logic_weight

        breakdown['logic_score'] = logic_score
        breakdown['logic_reward'] = logic_reward

        # === NOVELTY ===
        novelty_score = self.score_novelty(response, think_content)
        novelty_reward = novelty_score * w.novelty_weight

        breakdown['novelty_score'] = novelty_score
        breakdown['novelty_reward'] = novelty_reward

        # === TOTAL REWARD ===
        total_reward = citation_reward + format_reward + logic_reward + novelty_reward
        breakdown['total_reward'] = total_reward

        return total_reward, breakdown

    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        valid_pmids_list: List[Set[str]]
    ) -> List[float]:
        """
        Batch compute rewards for GRPO training.

        Args:
            prompts: List of input prompts (unused but kept for API compatibility)
            responses: List of model responses
            valid_pmids_list: List of valid PMID sets for each prompt

        Returns:
            List of reward scores
        """
        rewards = []
        for response, valid_pmids in zip(responses, valid_pmids_list):
            reward, _ = self.compute_reward(response, valid_pmids)
            rewards.append(reward)
        return rewards


def test_reward_function():
    """Test the reward function with example inputs"""
    reward_fn = BioReasonerRewardFunction()

    # Test case 1: Perfect response
    response_good = """
    <think>
    Let me analyze these papers carefully. Paper A (PMID: 12345678) demonstrates
    a novel mechanism for protein folding. This suggests that the pathway
    identified in PMID: 23456789 could be leveraged for drug discovery.

    I hypothesize that combining the approaches from both papers would result
    in improved therapeutic outcomes. This could be validated through
    experimental testing of the proposed mechanism.
    </think>

    Based on my analysis, I propose a novel integration of the protein folding
    mechanism (PMID: 12345678) with the therapeutic pathway (PMID: 23456789).
    """

    valid_pmids = {'12345678', '23456789', '34567890'}

    reward, breakdown = reward_fn.compute_reward(response_good, valid_pmids)
    print("=== Good Response ===")
    print(f"Total Reward: {reward:.2f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    # Test case 2: Hallucinating response
    response_bad = """
    <think>
    The paper PMID: 99999999 clearly shows that this approach works.
    </think>

    According to PMID: 99999999 and PMID: 88888888, the results are conclusive.
    """

    reward, breakdown = reward_fn.compute_reward(response_bad, valid_pmids)
    print("\n=== Bad Response (Hallucinations) ===")
    print(f"Total Reward: {reward:.2f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    # Test case 3: No citations
    response_no_cite = """
    <think>
    This is an interesting topic. More research is needed.
    </think>

    The topic requires further study.
    """

    reward, breakdown = reward_fn.compute_reward(response_no_cite, valid_pmids)
    print("\n=== No Citations Response ===")
    print(f"Total Reward: {reward:.2f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    # Test case 4: Missing <think>
    response_no_think = """
    This paper (PMID: 12345678) is very interesting and suggests novel approaches.
    """

    reward, breakdown = reward_fn.compute_reward(response_no_think, valid_pmids)
    print("\n=== Missing <think> Response ===")
    print(f"Total Reward: {reward:.2f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_reward_function()
