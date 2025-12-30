-- Database schema for rubric rewards training
-- Extends polymax-synthesizer/papers.db

-- Research triplets extracted from papers (Phase 1)
CREATE TABLE IF NOT EXISTS research_triplets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    sample_num INTEGER NOT NULL,  -- 1, 2, or 3
    research_goal TEXT NOT NULL,
    rubric JSON NOT NULL,  -- List of rubric items
    reference_solution TEXT NOT NULL,
    extractor_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    extraction_time_seconds REAL,
    word_counts JSON,  -- {goal: N, rubric_items: M, reference: K}
    UNIQUE(paper_id, sample_num),
    FOREIGN KEY(paper_id) REFERENCES papers(id)
);

-- Selected best triplets (Phase 2)
CREATE TABLE IF NOT EXISTS selected_triplets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triplet_id INTEGER NOT NULL UNIQUE,
    selection_score REAL NOT NULL,  -- Weighted total score
    scores JSON NOT NULL,  -- Individual criterion scores
    selection_reasoning TEXT,
    selector_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(triplet_id) REFERENCES research_triplets(id)
);

-- Training runs (Phase 3)
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT UNIQUE,
    stage INTEGER NOT NULL,  -- 1 or 2
    config JSON NOT NULL,
    start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_timestamp DATETIME,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    final_metrics JSON,
    checkpoint_path TEXT
);

-- Training metrics per epoch
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,
    rubric_satisfaction REAL,
    plan_length_avg REAL,
    loss REAL,
    learning_rate REAL,
    vram_used_mb INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES training_runs(id)
);

-- Generated plans during training (for analysis)
CREATE TABLE IF NOT EXISTS generated_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    triplet_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    generated_plan TEXT NOT NULL,
    thinking_section TEXT,
    rubric_scores JSON,  -- Per-item satisfaction
    length_violation BOOLEAN,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES training_runs(id),
    FOREIGN KEY(triplet_id) REFERENCES research_triplets(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_triplets_paper ON research_triplets(paper_id);
CREATE INDEX IF NOT EXISTS idx_selected_triplet ON selected_triplets(triplet_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON training_metrics(run_id, epoch);
CREATE INDEX IF NOT EXISTS idx_plans_run ON generated_plans(run_id, epoch);

-- Views for analysis
CREATE VIEW IF NOT EXISTS triplet_quality_stats AS
SELECT
    COUNT(*) as total_triplets,
    COUNT(DISTINCT paper_id) as unique_papers,
    AVG(json_extract(word_counts, '$.goal')) as avg_goal_words,
    AVG(json_extract(word_counts, '$.rubric_items')) as avg_rubric_items,
    AVG(json_extract(word_counts, '$.reference')) as avg_reference_words,
    AVG(extraction_time_seconds) as avg_extraction_time_sec
FROM research_triplets;

CREATE VIEW IF NOT EXISTS selection_quality_stats AS
SELECT
    COUNT(*) as total_selected,
    AVG(selection_score) as avg_score,
    MIN(selection_score) as min_score,
    MAX(selection_score) as max_score,
    AVG(json_extract(scores, '$.rubric_coverage')) as avg_rubric_coverage,
    AVG(json_extract(scores, '$.goal_clarity')) as avg_goal_clarity,
    AVG(json_extract(scores, '$.rubric_feasibility')) as avg_rubric_feasibility,
    AVG(json_extract(scores, '$.diversity')) as avg_diversity
FROM selected_triplets;

CREATE VIEW IF NOT EXISTS training_progress AS
SELECT
    tr.run_name,
    tr.stage,
    tr.status,
    tm.epoch,
    AVG(tm.rubric_satisfaction) as avg_rubric_satisfaction,
    AVG(tm.plan_length_avg) as avg_plan_length,
    AVG(tm.loss) as avg_loss,
    MAX(tm.vram_used_mb) as peak_vram_mb
FROM training_runs tr
JOIN training_metrics tm ON tr.id = tm.run_id
GROUP BY tr.id, tm.epoch
ORDER BY tr.id, tm.epoch;
