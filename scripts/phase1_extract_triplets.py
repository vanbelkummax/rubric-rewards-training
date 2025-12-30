#!/usr/bin/env python3
"""
Phase 1: Extract (goal, rubric, reference) triplets from papers.

Usage:
    # Pilot mode (20 papers for quality review)
    python phase1_extract_triplets.py --config configs/extraction_config.yaml --pilot-mode --num-papers 20

    # Full extraction (all papers)
    python phase1_extract_triplets.py --config configs/extraction_config.yaml --num-papers 830

Author: Max Van Belkum
Date: 2025-12-30
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.prompts import EXTRACTOR_PROMPT, EXTRACTION_EXAMPLES


class TripletExtractor:
    """Extract research triplets from papers using local LLM."""

    def __init__(self, config_path: str):
        """Initialize extractor with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model = self.config['model']
        self.ollama_url = self.config['ollama_url']
        self.samples_per_paper = self.config['extraction']['samples_per_paper']
        self.db_path = self.config['database']['path']

        # Setup logging
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "extraction.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Create output table if needed
        self._create_output_table()

    def _create_output_table(self):
        """Create research_triplets table in database."""
        schema = """
        CREATE TABLE IF NOT EXISTS research_triplets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER,
            sample_num INTEGER,
            research_goal TEXT NOT NULL,
            rubric JSON NOT NULL,
            reference_solution TEXT NOT NULL,
            extractor_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            extraction_time_seconds REAL,
            word_counts JSON,
            UNIQUE(paper_id, sample_num)
        )
        """
        self.conn.execute(schema)
        self.conn.commit()
        self.logger.info("Created/verified research_triplets table")

    def get_papers_to_process(self, num_papers: int, pilot_mode: bool = False) -> List[Dict]:
        """Get papers from database."""
        query = """
        SELECT p.id, p.title, p.pmid, p.year, p.journal,
               prof.name as professor
        FROM papers p
        JOIN professors prof ON p.professor_id = prof.id
        WHERE p.id NOT IN (
            SELECT DISTINCT paper_id FROM research_triplets
        )
        ORDER BY p.year DESC, p.pmid DESC
        LIMIT ?
        """

        cursor = self.conn.execute(query, (num_papers,))
        papers = [dict(row) for row in cursor.fetchall()]

        self.logger.info(f"Found {len(papers)} papers to process")
        if pilot_mode:
            self.logger.info(f"PILOT MODE: Processing first {len(papers)} papers for quality review")

        return papers

    def get_paper_content(self, paper_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Get paper summary and full text from paper_extractions table."""
        # Try to get summary
        summary_query = """
        SELECT content FROM paper_summaries
        WHERE paper_id = ?
        LIMIT 1
        """
        cursor = self.conn.execute(summary_query, (paper_id,))
        summary_row = cursor.fetchone()
        summary = summary_row['content'] if summary_row else None

        # Try to get full text from extractions (methods section preferred)
        fulltext_query = """
        SELECT section, content FROM paper_extractions
        WHERE paper_id = ?
        ORDER BY
            CASE
                WHEN section LIKE '%method%' THEN 1
                WHEN section LIKE '%result%' THEN 2
                ELSE 3
            END
        LIMIT 5
        """
        cursor = self.conn.execute(fulltext_query, (paper_id,))
        fulltext_rows = cursor.fetchall()

        if fulltext_rows:
            full_text = "\n\n".join([f"[{row['section']}]\n{row['content']}" for row in fulltext_rows])
        else:
            full_text = None

        return summary, full_text

    def call_ollama(self, prompt: str, timeout: int = 180) -> Optional[str]:
        """Call Ollama API with retry logic."""
        max_attempts = self.config['retry']['max_attempts']
        backoff = self.config['retry']['backoff']

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    f"{self.ollama_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 4096
                    },
                    timeout=timeout
                )
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content']

            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt+1}/{max_attempts}")
                if attempt < max_attempts - 1:
                    time.sleep(backoff * (attempt + 1))

            except Exception as e:
                self.logger.error(f"Error calling Ollama: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                else:
                    return None

        return None

    def extract_triplets_from_paper(self, paper: Dict) -> Optional[List[Dict]]:
        """Extract triplets from a single paper."""
        paper_id = paper['id']

        # Get paper content
        summary, full_text = self.get_paper_content(paper_id)

        if not summary and not full_text:
            self.logger.warning(f"No content found for paper {paper_id}: {paper['title']}")
            return None

        # Format prompt
        prompt = EXTRACTOR_PROMPT.format(
            num_samples=self.samples_per_paper,
            title=paper['title'],
            authors=paper.get('authors', 'Unknown'),
            year=paper.get('year', 'Unknown'),
            journal=paper.get('journal', 'Unknown'),
            summary=summary or "No summary available",
            full_text=full_text[:3000] if full_text else "No full text available"  # Truncate if too long
        )

        # Call LLM
        start_time = datetime.now()
        response = self.call_ollama(prompt)
        extraction_time = (datetime.now() - start_time).total_seconds()

        if not response:
            self.logger.error(f"Failed to get response for paper {paper_id}")
            return None

        # Parse JSON response
        try:
            # Try to extract JSON from response (handles cases where LLM adds text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            samples = data.get('samples', [])

            if len(samples) != self.samples_per_paper:
                self.logger.warning(
                    f"Expected {self.samples_per_paper} samples, got {len(samples)} for paper {paper_id}"
                )

            # Add metadata
            for sample in samples:
                sample['extraction_time_seconds'] = extraction_time
                sample['word_counts'] = {
                    'goal': len(sample['research_goal'].split()),
                    'rubric_items': len(sample['rubric']),
                    'reference': len(sample['reference_solution'].split())
                }

            return samples

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON for paper {paper_id}: {e}")
            self.logger.debug(f"Raw response: {response[:500]}")
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error parsing response for paper {paper_id}: {e}")
            return None

    def save_triplets(self, paper_id: int, samples: List[Dict]):
        """Save extracted triplets to database."""
        for i, sample in enumerate(samples, 1):
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO research_triplets
                    (paper_id, sample_num, research_goal, rubric, reference_solution,
                     extraction_time_seconds, word_counts)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper_id,
                    i,
                    sample['research_goal'],
                    json.dumps(sample['rubric']),
                    sample['reference_solution'],
                    sample['extraction_time_seconds'],
                    json.dumps(sample['word_counts'])
                ))
                self.conn.commit()

            except Exception as e:
                self.logger.error(f"Failed to save triplet {i} for paper {paper_id}: {e}")

    def run_extraction(self, num_papers: int, pilot_mode: bool = False) -> Dict:
        """Run extraction pipeline."""
        papers = self.get_papers_to_process(num_papers, pilot_mode)

        if not papers:
            self.logger.info("No papers to process")
            return {"papers_processed": 0, "triplets_extracted": 0}

        stats = {
            "papers_processed": 0,
            "triplets_extracted": 0,
            "papers_failed": 0,
            "avg_extraction_time": 0,
            "total_time": 0
        }

        start_time = datetime.now()

        for paper in tqdm(papers, desc="Extracting triplets"):
            triplets = self.extract_triplets_from_paper(paper)

            if triplets:
                self.save_triplets(paper['id'], triplets)
                stats['papers_processed'] += 1
                stats['triplets_extracted'] += len(triplets)
            else:
                stats['papers_failed'] += 1

        stats['total_time'] = (datetime.now() - start_time).total_seconds()
        if stats['papers_processed'] > 0:
            stats['avg_extraction_time'] = stats['total_time'] / stats['papers_processed']

        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Papers processed: {stats['papers_processed']}/{len(papers)}")
        self.logger.info(f"Triplets extracted: {stats['triplets_extracted']}")
        self.logger.info(f"Papers failed: {stats['papers_failed']}")
        self.logger.info(f"Avg time per paper: {stats['avg_extraction_time']:.1f}s")
        self.logger.info(f"Total time: {stats['total_time']/60:.1f} minutes")

        if pilot_mode:
            self.logger.info("\n" + "="*60)
            self.logger.info("PILOT MODE: Review triplets in outputs/triplets/pilot/")
            self.logger.info("Check quality before proceeding to full extraction")
            self.logger.info("="*60)

        return stats

    def export_triplets_to_json(self, output_dir: str = "outputs/triplets"):
        """Export extracted triplets to JSON for review."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        query = """
        SELECT
            rt.id,
            rt.paper_id,
            rt.sample_num,
            rt.research_goal,
            rt.rubric,
            rt.reference_solution,
            rt.word_counts,
            p.title,
            p.year,
            prof.name as professor
        FROM research_triplets rt
        JOIN papers p ON rt.paper_id = p.id
        JOIN professors prof ON p.professor_id = prof.id
        ORDER BY rt.paper_id, rt.sample_num
        """

        cursor = self.conn.execute(query)
        triplets = []

        for row in cursor.fetchall():
            triplet = {
                "triplet_id": row['id'],
                "paper_id": row['paper_id'],
                "paper_title": row['title'],
                "paper_year": row['year'],
                "professor": row['professor'],
                "sample_num": row['sample_num'],
                "research_goal": row['research_goal'],
                "rubric": json.loads(row['rubric']),
                "reference_solution": row['reference_solution'],
                "word_counts": json.loads(row['word_counts'])
            }
            triplets.append(triplet)

        output_file = output_path / "all_triplets.json"
        with open(output_file, 'w') as f:
            json.dump(triplets, f, indent=2)

        self.logger.info(f"Exported {len(triplets)} triplets to {output_file}")

        # Also export summary stats
        stats_file = output_path / "extraction_stats.json"
        stats = self._compute_stats(triplets)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Exported stats to {stats_file}")

    def _compute_stats(self, triplets: List[Dict]) -> Dict:
        """Compute statistics on extracted triplets."""
        goal_lengths = [t['word_counts']['goal'] for t in triplets]
        rubric_counts = [t['word_counts']['rubric_items'] for t in triplets]
        ref_lengths = [t['word_counts']['reference'] for t in triplets]

        return {
            "total_triplets": len(triplets),
            "unique_papers": len(set(t['paper_id'] for t in triplets)),
            "goal_length_stats": {
                "mean": sum(goal_lengths) / len(goal_lengths),
                "min": min(goal_lengths),
                "max": max(goal_lengths)
            },
            "rubric_item_stats": {
                "mean": sum(rubric_counts) / len(rubric_counts),
                "min": min(rubric_counts),
                "max": max(rubric_counts),
                "distribution": {
                    str(i): rubric_counts.count(i)
                    for i in range(min(rubric_counts), max(rubric_counts)+1)
                }
            },
            "reference_length_stats": {
                "mean": sum(ref_lengths) / len(ref_lengths),
                "min": min(ref_lengths),
                "max": max(ref_lengths)
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Extract research triplets from papers")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--num-papers", type=int, default=830, help="Number of papers to process")
    parser.add_argument("--pilot-mode", action="store_true", help="Run in pilot mode for quality review")
    parser.add_argument("--export", action="store_true", help="Export triplets to JSON after extraction")

    args = parser.parse_args()

    if args.pilot_mode and args.num_papers > 20:
        print("WARNING: Pilot mode recommended for ≤20 papers. Proceeding anyway...")

    extractor = TripletExtractor(args.config)
    stats = extractor.run_extraction(args.num_papers, args.pilot_mode)

    if args.export or args.pilot_mode:
        extractor.export_triplets_to_json()

    # Return exit code based on success rate
    success_rate = stats['papers_processed'] / (stats['papers_processed'] + stats['papers_failed'])
    if success_rate < 0.8:
        print(f"\nWARNING: Success rate {success_rate:.1%} < 80%")
        print("Review extraction.log for errors")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    import time  # Import here to avoid circular imports
    main()
