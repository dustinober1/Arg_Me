"""
Argument Hygiene & Procedural Noise Filtering Module for args.me.

Disentangles real substantive rhetorical arguments from platform procedural chatter,
forfeits, turn declarations, and typographical anomalies common to scraped online debates.

Features:
1. Fast rule-based heuristic scanner (word counts, procedural regex patterns, turn-order markers).
2. DSPy ArgumentSubstanceSignature for borderline case arbitration.
3. Corpus hygiene auditor providing noise breakdown analytics and pure-substance subsets.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


# Common procedural patterns from Debate.org and online debate portals
PROCEDURAL_REGEX_PATTERNS = [
    r"^\s*(extend|forfeit|pass|accept|concede|forfeited)\.?\s*$",
    r"^\s*i\s+(am|accept)\s+(con|pro)\b",
    r"\b(pro|con)\s+should\s+start\s+(the\s+argument\s+)?first\b",
    r"^\s*(this\s+is\s+going\s+well|no\s+argument|good\s+luck|waiting\s+for\s+opponent)\b",
    r"^\s*(\.|\?|\!|:|;|-|\/|l|k)+\s*$",
    r"\bplease\s+(vote|judge)\s+(on|based\s+on)\s+(conduct|forfeits?)\b",
    r"^\s*round\s+\d+\s*:\s*(extend|forfeit|pass)\b",
]

COMPILED_PROCEDURAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROCEDURAL_REGEX_PATTERNS]


def classify_argument_hygiene(premise_text: str) -> Dict[str, Any]:
    """
    Classify whether a premise text contains a substantive rhetorical argument
    or is merely procedural noise / platform chatter.

    Returns:
        dict: {
            "is_substantive": bool,
            "noise_category": Optional[str],  # 'too_short', 'procedural_chatter', 'forfeit', None
            "word_count": int,
            "char_count": int,
        }
    """
    text = str(premise_text).strip()
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # 1. Extreme brevity check
    if word_count < 4 or char_count < 15:
        return {
            "is_substantive": False,
            "noise_category": "too_short",
            "word_count": word_count,
            "char_count": char_count,
        }

    # 2. Known procedural expressions
    for pattern in COMPILED_PROCEDURAL_PATTERNS:
        if pattern.search(text):
            category = "forfeit" if "forfeit" in text.lower() or "extend" in text.lower() else "procedural_chatter"
            return {
                "is_substantive": False,
                "noise_category": category,
                "word_count": word_count,
                "char_count": char_count,
            }

    # 3. Ratio of punctuation to alphanumeric characters (typographical spam)
    alpha_chars = sum(c.isalnum() for c in text)
    if alpha_chars < 8 or (alpha_chars / max(1, char_count)) < 0.4:
        return {
            "is_substantive": False,
            "noise_category": "typographical_noise",
            "word_count": word_count,
            "char_count": char_count,
        }

    return {
        "is_substantive": True,
        "noise_category": None,
        "word_count": word_count,
        "char_count": char_count,
    }


def filter_corpus_hygiene(df: pd.DataFrame, text_column: str = "premise_text") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partition a debate argument DataFrame into clean substantive arguments and noise records.

    Returns:
        (clean_df, noise_df, audit_stats)
    """
    results = [classify_argument_hygiene(t) for t in df[text_column]]
    
    hygiene_df = df.copy()
    hygiene_df["is_substantive"] = [r["is_substantive"] for r in results]
    hygiene_df["noise_category"] = [r["noise_category"] for r in results]
    hygiene_df["word_count"] = [r["word_count"] for r in results]

    clean_df = hygiene_df[hygiene_df["is_substantive"]].reset_index(drop=True)
    noise_df = hygiene_df[~hygiene_df["is_substantive"]].reset_index(drop=True)

    n_total = len(df)
    n_clean = len(clean_df)
    n_noise = len(noise_df)

    audit_stats = {
        "total_records": n_total,
        "clean_records": n_clean,
        "noise_records": n_noise,
        "clean_percentage": round((n_clean / max(1, n_total)) * 100.0, 2),
        "noise_percentage": round((n_noise / max(1, n_total)) * 100.0, 2),
        "noise_breakdown": noise_df["noise_category"].value_counts().to_dict(),
    }

    return clean_df, noise_df, audit_stats
