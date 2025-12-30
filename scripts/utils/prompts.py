"""
Prompts for triplet extraction and selection.
Based on arxiv 2512.23707 methodology.
"""

EXTRACTOR_PROMPT = """You are extracting training data for an AI co-scientist from a research paper.

Your task: Generate {num_samples} diverse research goals from the paper below.

REQUIREMENTS FOR EACH RESEARCH GOAL:
1. Target a KEY INSIGHT or contribution from the paper
2. Include SPECIFIC CONSTRAINTS and preferences stated in the paper
3. Be OPEN-ENDED enough to require a multi-step research plan
4. Be CONCRETE and actionable (not vague or theoretical)

For EACH goal, also generate:
- A GOAL-SPECIFIC GRADING RUBRIC (5-8 items)
  - Each item must be MEASURABLE and SPECIFIC
  - Use action verbs: "Must achieve...", "Should include...", "Must outperform..."
  - No vague items like "Should be good quality"
- A REFERENCE SOLUTION (the actual approach taken in the paper)
  - Describe what the paper actually did in 200-500 words
  - Include methods, key techniques, validation approach

PAPER INFORMATION:
Title: {title}
Authors: {authors}
Year: {year}
Journal: {journal}

PAPER SUMMARY:
{summary}

PAPER FULL TEXT (Methods section):
{full_text}

OUTPUT FORMAT (strict JSON):
{{
  "samples": [
    {{
      "research_goal": "Develop a method to...",
      "rubric": [
        "Must address challenge X from the introduction",
        "Should outperform baseline Y by Z%",
        "Must handle edge case W",
        "Should be computationally efficient (<T minutes)",
        "Must preserve property P"
      ],
      "reference_solution": "The paper proposes METHOD, which uses TECHNIQUE1 and TECHNIQUE2. They validate on DATASET with METRICS..."
    }}
  ]
}}

CRITICAL: Output ONLY valid JSON. No explanation before or after.
"""

SELECTOR_PROMPT = """You are selecting the HIGHEST-QUALITY training sample from {num_candidates} candidates.

SCORING CRITERIA (0-10 each):
1. **Rubric Coverage** (30% weight)
   - Do the rubric items comprehensively capture ALL essential requirements?
   - Are rubric items specific and measurable (not vague)?
   - Do they match the research goal's scope?

2. **Goal Clarity** (25% weight)
   - Is the research goal clear and unambiguous?
   - Can someone implement it from this description?
   - No hallucinations or fabricated details?

3. **Rubric Feasibility** (25% weight)
   - Can rubric items be automatically evaluated by an LLM?
   - Are they objective (not subjective like "should be elegant")?
   - Do they avoid requiring domain knowledge not in the paper?

4. **Diversity & Uniqueness** (20% weight)
   - Does this sample add unique value vs other papers?
   - Captures a distinct research pattern or approach?
   - Not redundant with common ML patterns?

CANDIDATES:
{candidates_json}

TASK:
1. Score each candidate (0-10) on all 4 criteria
2. Compute weighted total score
3. Select the highest-scoring candidate
4. Provide 2-3 sentence reasoning

OUTPUT FORMAT (strict JSON):
{{
  "selected_index": 0,  # Index of best candidate (0, 1, or 2)
  "scores": [
    {{
      "rubric_coverage": 8.5,
      "goal_clarity": 9.0,
      "rubric_feasibility": 7.5,
      "diversity": 8.0,
      "total": 8.3
    }},
    ...
  ],
  "reasoning": "Candidate 0 has the most specific and measurable rubric items, with clear connections to the paper's contributions..."
}}

CRITICAL: Output ONLY valid JSON. No explanation before or after.
"""

GRADING_PROMPT = """You are grading a research plan against a specific rubric item.

RUBRIC ITEM:
{rubric_item}

RESEARCH PLAN:
{plan}

GENERAL GUIDELINES (ALL must be satisfied):
1. Plan must DIRECTLY address the rubric item (not tangentially)
2. Plan must be SPECIFIC (no vague statements like "we will optimize")
3. Plan must be ACTIONABLE (clear steps, not just ideas)
4. Plan must be COMPLETE for this rubric item (not missing key components)
5. Plan must be FEASIBLE (not requiring impossible resources)
6. Plan must be COHERENT (logically consistent)
7. Plan must be GROUNDED (references specific techniques/methods)

TASK:
Does the plan satisfy this rubric item AND all 7 general guidelines?

Answer with ONLY:
- "YES" if ALL requirements are met
- "NO" if ANY requirement is not met

Do not explain. Output ONLY "YES" or "NO".
"""

LENGTH_VIOLATION_PROMPT = """You are checking if a research plan violates length constraints.

PLAN (within <plan> tags):
{plan_with_tags}

CONSTRAINTS:
- Thinking section (before <plan>): UNLIMITED (can be any length)
- Plan section (within <plan>...</plan>): MAX {max_words} words

TASK:
Count words in the <plan> section only. Ignore thinking section.

OUTPUT FORMAT (strict JSON):
{{
  "plan_word_count": 523,
  "max_allowed": 500,
  "violates_length": true,
  "violation_words": 23
}}

CRITICAL: Output ONLY valid JSON.
"""

# Few-shot examples for extraction (high quality)
EXTRACTION_EXAMPLES = [
    {
        "paper": "Img2ST: Imaging to Spatial Transcriptomics Mapping",
        "goal": "Develop a deep learning method to predict spatial gene expression at subcellular resolution from H&E histology images",
        "rubric": [
            "Must achieve SSIM >0.55 on Visium HD 2μm resolution data",
            "Should outperform existing methods (Hist2ST, ST-Net) by >5%",
            "Must handle tissue heterogeneity (tumor, stroma, immune)",
            "Should preserve spatial gene correlation structure (Moran's I >0.3)",
            "Must be computationally efficient (<10 minutes per tissue section)",
            "Should generalize across tissue types (CRC, breast, lung)"
        ],
        "reference": "The paper proposes Img2ST-Net, combining a Prov-GigaPath encoder for feature extraction with a Hist2ST decoder for gene expression prediction. They use MSE loss with spatial regularization, train on 3 CRC patients with Visium HD data, and achieve SSIM 0.5699 at 2μm resolution."
    },
    {
        "paper": "SIID: Spatial Imputation with Interpretable Deconvolution",
        "goal": "Create a method to simultaneously perform cell type deconvolution and gene imputation in spatial transcriptomics data",
        "rubric": [
            "Must decompose spatial spots into cell type proportions",
            "Should impute missing genes with correlation >0.7 to ground truth",
            "Must be interpretable (provide cell type signatures)",
            "Should handle both Visium (55μm) and Visium HD (2μm) resolutions",
            "Must preserve spatial patterns (Moran's I decrease <10%)",
            "Should work without requiring single-cell reference (optional)"
        ],
        "reference": "SIID uses non-negative matrix factorization with spatial smoothness constraints. It jointly learns cell type signatures and their spatial distributions, validated on mouse brain and human CRC data."
    }
]
