"""
Active Learning & Hard-Negative Mining Engine for args.me.

Streams debate arguments from the full 387,000-record Webis corpus (webis/args_me),
computes Shannon entropy and margin uncertainty distributions, and mines
hard dialectical negatives to curate an active learning curriculum.

Theoretical Foundations:
- Shannon Entropy: H(x) = -sum_y P(y|x) log2 P(y|x)
- Margin Sampling: M(x) = |P(PRO|x) - P(CON|x)|
- Hard Negatives: Disagreements between high-confidence model predictions and gold labels
  representing rhetorical irony, double negations, and opponent quotations.
"""

import os
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Iterator
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset_loader import get_args_dataset, flatten_argument_record
from src.argument_hygiene import classify_argument_hygiene
from src.dspy_mipro_copro import NimbleStancePredictor


def compute_uncertainty_metrics(pro_prob: float, con_prob: float) -> Dict[str, float]:
    """
    Compute information-theoretic uncertainty metrics for a binary probability distribution.
    """
    p = float(np.clip(pro_prob, 1e-7, 1.0 - 1e-7))
    q = float(np.clip(con_prob, 1e-7, 1.0 - 1e-7))

    # Normalize if slightly off
    total = p + q
    p /= total
    q /= total

    # Shannon Entropy in bits (max = 1.0 for uniform binary distribution)
    entropy = - (p * math.log2(p) + q * math.log2(q))
    margin = abs(p - q)
    least_confidence = 1.0 - max(p, q)

    return {
        "entropy": round(float(entropy), 4),
        "margin": round(float(margin), 4),
        "least_confidence": round(float(least_confidence), 4),
    }


def categorize_hard_case(
    gold_stance: str,
    pred_stance: str,
    pro_prob: float,
    confidence: float,
    premise_text: str,
) -> str:
    """
    Categorize mined hard arguments into a dialectic error taxonomy.
    """
    text_lower = premise_text.lower()
    is_disagreement = (gold_stance != pred_stance) and (gold_stance in ["PRO", "CON"])
    margin = abs(pro_prob - (1.0 - pro_prob))

    # 1. High-Entropy Dilemma (uncertainty)
    if margin < 0.15:
        return "high_entropy_dilemma"

    # 2. Adversarial Quoting
    if ("opponent" in text_lower or "pro argues" in text_lower or "con argues" in text_lower or "claims that" in text_lower):
        return "adversarial_quoting_trap"

    # 3. High-Confidence Contradiction / Sarcasm
    if is_disagreement and confidence >= 0.70:
        return "high_confidence_contradiction"

    # 4. Concession / Multi-Clause Complexity
    if "concede" in text_lower or "admit" in text_lower or "however" in text_lower or "although" in text_lower:
        return "concession_complexity"

    return "dialectical_anomaly" if is_disagreement else "standard_example"


class ActiveLearningHardNegativeMiner:
    """
    Streams across args.me, evaluates arguments with decision models,
    and mines high-entropy and high-confidence hard negatives.
    """

    def __init__(
        self,
        model_name: str = "nimble",
        ollama_base_url: str = "http://localhost:11434",
        min_word_count: int = 10,
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.min_word_count = min_word_count
        self.predictor = NimbleStancePredictor(
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            custom_instruction="Analyze the author's ultimate thesis relative to the debate motion, accounting for adversary citations.",
        )

    def mine_from_dataframe(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scan a DataFrame of arguments, score uncertainty metrics, and extract hard-negative curriculum.
        """
        eval_df = df.copy()
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

        results = []
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Active Learning Uncertainty Scan"):
            topic = str(row.get("conclusion", ""))
            premise = str(row.get("premise_text", ""))
            gold = str(row.get("stance", "UNKNOWN")).strip().upper()

            # Filter trivial noise
            hygiene = classify_argument_hygiene(premise)
            if not hygiene["is_substantive"]:
                continue

            pred = self.predictor.forward(topic, premise)
            uncert = compute_uncertainty_metrics(pred.pro_prob, pred.con_prob)
            category = categorize_hard_case(gold, pred.stance, pred.pro_prob, pred.confidence, premise)

            results.append({
                "conclusion": topic,
                "premise_text": premise,
                "gold_stance": gold,
                "pred_stance": pred.stance,
                "is_correct": (gold == pred.stance),
                "pro_prob": pred.pro_prob,
                "con_prob": pred.con_prob,
                "confidence": pred.confidence,
                "entropy": uncert["entropy"],
                "margin": uncert["margin"],
                "least_confidence": uncert["least_confidence"],
                "taxonomy_category": category,
                "word_count": hygiene["word_count"],
            })

        mined_df = pd.DataFrame(results)

        # Segment into hard cases: High Entropy (entropy >= 0.85) OR Model Disagreements
        hard_cases = mined_df[(mined_df["entropy"] >= 0.85) | (~mined_df["is_correct"])].copy()
        hard_cases = hard_cases.sort_values(["entropy", "confidence"], ascending=[False, False]).reset_index(drop=True)

        stats = {
            "total_screened": len(mined_df),
            "hard_cases_identified": len(hard_cases),
            "hard_case_percentage": round((len(hard_cases) / max(1, len(mined_df))) * 100.0, 2),
            "mean_entropy": round(float(mined_df["entropy"].mean()), 4),
            "taxonomy_breakdown": hard_cases["taxonomy_category"].value_counts().to_dict(),
        }

        return hard_cases, stats

    def stream_and_mine(
        self,
        max_arguments: int = 500,
        streaming_split: str = "train",
        output_parquet: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Stream arguments directly from Hugging Face hub (webis/args_me) and mine hard negatives.
        """
        print(f"Streaming from Hugging Face webis/args_me (target: {max_arguments} premises)...")
        dataset = get_args_dataset(streaming=True, split=streaming_split)

        collected = []
        for record in dataset:
            premises = flatten_argument_record(record)
            for p in premises:
                if len(str(p.get("premise_text", "")).split()) >= self.min_word_count:
                    collected.append(p)
                    if len(collected) >= max_arguments:
                        break
            if len(collected) >= max_arguments:
                break

        print(f"Collected {len(collected)} substantial premises from stream. Initiating active uncertainty scan...")
        df_stream = pd.DataFrame(collected)
        hard_cases, stats = self.mine_from_dataframe(df_stream)

        if output_parquet:
            os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
            hard_cases.to_parquet(output_parquet, index=False)
            print(f"Mined hard-negative curriculum saved to: {output_parquet}")

        return hard_cases, stats
