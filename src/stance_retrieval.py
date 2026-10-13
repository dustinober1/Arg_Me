"""
End-to-End Stance-Aware Argument Retrieval Engine for args.me.

Implements the Wachsmuth et al. (2017) dual-column debate search architecture:
1. Lexical / Semantic Retrieval: Indexes argument conclusions and premises, retrieving candidate arguments for a query.
2. Decision Model Re-Ranking & Stance Assignment: Bespoke Labs Nimble 9B evaluates stance (PRO vs. CON) and confidence.
3. Dual-Column Aggregation: Organizes arguments into affirmative (PRO) and opposing (CON) columns ranked by calibrated score.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.dspy_mipro_copro import NimbleStancePredictor


class StanceAwareArgumentSearchEngine:
    """
    Search and retrieval engine indexing debate arguments from args.me,
    scoring query relevance, and partitioning results into PRO and CON debate columns.
    """

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        model_name: str = "nimble",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.model_name = model_name
        self.classifier = NimbleStancePredictor(
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            custom_instruction="Analyze the author's ultimate thesis relative to the debate motion, accounting for adversary citations.",
        )

        print(f"Indexing {len(self.corpus_df):,} debate arguments for retrieval...")
        self.vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
        # Build search text: combination of conclusion motion and premise text
        search_corpus = [
            f"{row.get('conclusion', '')} {row.get('premise_text', '')}"
            for _, row in self.corpus_df.iterrows()
        ]
        self.corpus_matrix = self.vectorizer.fit_transform(search_corpus)
        print("Corpus indexed successfully!")

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_relevance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Search corpus for query, re-rank with Nimble 9B, and partition into PRO and CON columns.
        """
        t0 = time.time()
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.corpus_matrix).flatten()

        top_indices = np.argsort(sim_scores)[::-1][: top_k * 3]
        candidate_records = []

        for idx in top_indices:
            rel = float(sim_scores[idx])
            if rel < min_relevance:
                continue
            row = self.corpus_df.iloc[idx]
            candidate_records.append({
                "index": int(idx),
                "conclusion": str(row.get("conclusion", "")),
                "premise_text": str(row.get("premise_text", "")),
                "gold_stance": str(row.get("stance", "UNKNOWN")),
                "relevance_score": rel,
            })

        # Classify stance using Nimble 9B
        classified_candidates = []
        for cand in candidate_records[: top_k * 2]:
            pred = self.classifier.forward(query, cand["premise_text"])
            cand["predicted_stance"] = pred.stance
            cand["confidence"] = pred.confidence
            cand["pro_prob"] = pred.pro_prob
            cand["con_prob"] = pred.con_prob
            # Composite rank score = relevance * (0.5 + confidence/2.0)
            cand["rank_score"] = cand["relevance_score"] * (0.5 + cand["confidence"] * 0.5)
            classified_candidates.append(cand)

        pro_args = [c for c in classified_candidates if c["predicted_stance"] == "PRO"]
        con_args = [c for c in classified_candidates if c["predicted_stance"] == "CON"]

        pro_args.sort(key=lambda x: x["rank_score"], reverse=True)
        con_args.sort(key=lambda x: x["rank_score"], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "query": query,
            "total_candidates": len(candidate_records),
            "pro_count": len(pro_args),
            "con_count": len(con_args),
            "elapsed_ms": round(elapsed_ms, 2),
            "pro_arguments": pro_args[:top_k],
            "con_arguments": con_args[:top_k],
        }

    def format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format dual-column search results for terminal / notebook rendering."""
        lines = [
            "=" * 80,
            f"STANCE-AWARE ARGUMENT SEARCH: \"{search_results['query']}\"",
            f"Retrieval Latency: {search_results['elapsed_ms']} ms | PRO Args: {search_results['pro_count']} | CON Args: {search_results['con_count']}",
            "=" * 80,
            "",
            "--- [PRO / AFFIRMATIVE COLUMN] ---",
        ]
        for i, arg in enumerate(search_results["pro_arguments"][:3], 1):
            lines.append(f"[{i}] Rank Score: {arg['rank_score']:.3f} | Conf: {arg['confidence']:.2f}")
            lines.append(f"    Topic: {arg['conclusion'][:60]}")
            lines.append(f"    Claim: {arg['premise_text'][:140]}...")
            lines.append("")

        lines.append("--- [CON / OPPOSING COLUMN] ---")
        for i, arg in enumerate(search_results["con_arguments"][:3], 1):
            lines.append(f"[{i}] Rank Score: {arg['rank_score']:.3f} | Conf: {arg['confidence']:.2f}")
            lines.append(f"    Topic: {arg['conclusion'][:60]}")
            lines.append(f"    Claim: {arg['premise_text'][:140]}...")
            lines.append("")

        return "\n".join(lines)
