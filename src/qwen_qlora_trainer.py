"""
Qwen QLoRA Stance Classifier Training & Evaluation Module.

Provides utilities for 4-bit NormalFloat (NF4) parameter-efficient fine-tuning
of Qwen 2.5/3.5 models on args.me debate stance classification, with support
for hard-negative curriculum integration and Google Colab GPU runtimes.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

SYSTEM_PROMPT = (
    "You are an expert dialectical stance classifier. Analyze the debate topic and the participant's premise. "
    "Output either [[PRO]] if the premise supports the resolution, or [[CON]] if the premise opposes or refutes "
    "the resolution, followed by a concise dialectical rationale."
)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def format_chat_record(conclusion: str, premise_text: str, stance: Optional[str] = None) -> List[Dict[str, str]]:
    """Format an argument into the standard chat conversation structure for Qwen."""
    user_content = f"Debate Motion: {conclusion}\nPremise: {premise_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if stance is not None:
        stance_clean = stance.strip().upper()
        assistant_content = f"Stance: [[{stance_clean}]]"
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def build_training_dataset(
    df: pd.DataFrame,
    hard_negatives_df: Optional[pd.DataFrame] = None,
    hard_negative_weight: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs a balanced train and test set from cleaned argument data,
    optionally upsampling mined hard negatives from the curriculum.
    """
    # Ensure binary stance
    clean_df = df[df["stance"].isin(["PRO", "CON"])].copy()

    # Stratified split
    train_df, test_df = train_test_split(
        clean_df, test_size=test_size, random_state=random_state, stratify=clean_df["stance"]
    )
    train_df = train_df.copy()

    # Integrate hard negative curriculum if supplied
    if hard_negatives_df is not None and not hard_negatives_df.empty:
        hn_clean = hard_negatives_df[hard_negatives_df["stance"].isin(["PRO", "CON"])].copy()
        # Replicate hard negatives to emphasize difficult decision boundaries
        replicated_hn = pd.concat([hn_clean] * hard_negative_weight, ignore_index=True)
        train_df = pd.concat([train_df, replicated_hn], ignore_index=True).sample(
            frac=1.0, random_state=random_state
        ).reset_index(drop=True)

    return train_df, test_df


def prepare_hf_conversations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe records into Hugging Face chat-formatted dictionaries."""
    records = []
    for _, row in df.iterrows():
        messages = format_chat_record(
            conclusion=str(row["conclusion"]),
            premise_text=str(row["premise_text"]),
            stance=str(row["stance"]),
        )
        records.append({"messages": messages})
    return records


def parse_stance_prediction(text: str) -> str:
    """Robustly extract PRO or CON from model generated response."""
    match = re.search(r"\[\[(PRO|CON)\]\]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback heuristic
    upper_text = text.upper()
    has_pro = "PRO" in upper_text
    has_con = "CON" in upper_text
    if has_pro and not has_con:
        return "PRO"
    if has_con and not has_pro:
        return "CON"
    return "UNKNOWN"


def compute_stance_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute classification metrics for binary stance predictions."""
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p in ["PRO", "CON"]]
    if not valid_pairs:
        return {"accuracy": 0.0, "pro_f1": 0.0, "con_f1": 0.0, "macro_f1": 0.0}

    y_t = [t for t, _ in valid_pairs]
    y_p = [p for _, p in valid_pairs]

    acc = accuracy_score(y_t, y_p)
    p_scores, r_scores, f_scores, _ = precision_recall_fscore_support(
        y_t, y_p, labels=["PRO", "CON"], average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f_scores))

    return {
        "accuracy": float(acc),
        "pro_precision": float(p_scores[0]),
        "pro_recall": float(r_scores[0]),
        "pro_f1": float(f_scores[0]),
        "con_precision": float(p_scores[1]),
        "con_recall": float(r_scores[1]),
        "con_f1": float(f_scores[1]),
        "macro_f1": macro_f1,
        "valid_eval_rate": float(len(valid_pairs) / len(y_true)),
    }
