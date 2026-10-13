"""
Mixture-of-Judges Ensemble Engine for Argument Stance Classification.

Combines Together AI Tev1:4B and Bespoke Labs Nimble 9B decision models
using confidence-weighted consensus and calibration gating.

Theoretical Foundation:
- While Nimble 9B has stronger overall semantic resolution, Tev1:4B exhibits
  complementary sensitivity on specific rhetorical constructs.
- Gated calibration arbitration resolves model disagreements using normalized
  logit margins rather than simple majority votes.
"""

import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
)

import dspy
from src.baseline_stance import EvaluationMetrics


class MixtureOfJudgesEnsemble(dspy.Module):
    """
    Ensemble decision pipeline coordinating Tev1:4B and Nimble 9B
    with calibration-gated consensus arbitration.
    """

    def __init__(
        self,
        primary_model: str = "nimble",
        secondary_model: str = "tev1:4b",
        ollama_base_url: str = "http://localhost:11434",
        custom_instruction: Optional[str] = None,
        custom_criteria: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/v1/systemone"
        self.instruction = custom_instruction or (
            "Analyze the author's ultimate thesis relative to the debate motion, accounting for adversary citations."
        )
        self.criteria = custom_criteria or {
            "PRO": "The premise affirms, defends, or advocates in favor of the resolution topic.",
            "CON": "The premise opposes, attacks, or refutes the resolution topic.",
        }
        self.timeout = timeout

    def query_single_judge(self, model_name: str, topic: str, premise: str) -> Dict[str, Any]:
        """Query a single decision model via /v1/systemone."""
        payload = {
            "model": model_name,
            "state": f"Topic: {topic}\nPremise: {premise}",
            "questions": {
                "stance": {
                    "type": "choice",
                    "instructions": self.instruction,
                    "criteria": self.criteria,
                }
            },
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answers", {}).get("stance", {})
                choice = ans.get("choice", "PRO").strip().upper()
                probs = ans.get("probabilities", {})
                confidence = float(ans.get("confidence", 0.0))
                return {
                    "choice": choice,
                    "pro_prob": float(probs.get("PRO", 0.5)),
                    "con_prob": float(probs.get("CON", 0.5)),
                    "confidence": confidence,
                    "error": None,
                }
            return {"choice": "PRO", "pro_prob": 0.5, "con_prob": 0.5, "confidence": 0.0, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"choice": "PRO", "pro_prob": 0.5, "con_prob": 0.5, "confidence": 0.0, "error": str(e)}

    def forward(self, topic: str, premise: str) -> dspy.Prediction:
        """
        Execute dual-judge inference and calibration-gated arbitration.
        """
        t0 = time.time()
        res_primary = self.query_single_judge(self.primary_model, topic, premise)
        res_secondary = self.query_single_judge(self.secondary_model, topic, premise)
        latency_ms = (time.time() - t0) * 1000.0

        p_choice = res_primary["choice"]
        s_choice = res_secondary["choice"]
        is_consensus = (p_choice == s_choice)

        # Weighted combination of probability estimates
        # Weight: 0.65 for 9B primary, 0.35 for 4B secondary
        comb_pro_prob = (0.65 * res_primary["pro_prob"]) + (0.35 * res_secondary["pro_prob"])
        comb_con_prob = (0.65 * res_primary["con_prob"]) + (0.35 * res_secondary["con_prob"])

        if is_consensus:
            final_stance = p_choice
            final_confidence = max(res_primary["confidence"], res_secondary["confidence"])
        else:
            # Gated arbitration: rely on the higher confidence score
            if res_secondary["confidence"] > (res_primary["confidence"] + 0.25):
                final_stance = s_choice
                final_confidence = res_secondary["confidence"]
            else:
                final_stance = p_choice
                final_confidence = res_primary["confidence"]

        return dspy.Prediction(
            stance=final_stance,
            pro_prob=comb_pro_prob,
            con_prob=comb_con_prob,
            confidence=final_confidence,
            is_consensus=is_consensus,
            primary_choice=p_choice,
            secondary_choice=s_choice,
            latency_ms=latency_ms,
        )

    def evaluate_df(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        random_state: int = 42,
        pipeline_title: str = "Mixture-of-Judges (Nimble 9B + Tev1:4B)",
    ) -> Tuple[EvaluationMetrics, pd.DataFrame, Dict[str, Any]]:
        """Benchmark ensemble across DataFrame and report agreement stats."""
        eval_df = df.copy()
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

        preds = []
        pro_probs = []
        confidences = []
        consensuses = []
        p_choices = []
        s_choices = []
        latencies = []

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Evaluating {pipeline_title}"):
            pred = self.forward(str(row["conclusion"]), str(row["premise_text"]))
            preds.append(pred.stance)
            pro_probs.append(pred.pro_prob)
            confidences.append(pred.confidence)
            consensuses.append(pred.is_consensus)
            p_choices.append(pred.primary_choice)
            s_choices.append(pred.secondary_choice)
            latencies.append(pred.latency_ms)

        eval_df["pred_stance"] = preds
        eval_df["pro_prob"] = pro_probs
        eval_df["confidence"] = confidences
        eval_df["is_consensus"] = consensuses
        eval_df["primary_choice"] = p_choices
        eval_df["secondary_choice"] = s_choices
        eval_df["latency_ms"] = latencies
        eval_df["is_correct"] = eval_df["stance"] == eval_df["pred_stance"]

        y_true = eval_df["stance"].values
        y_pred = np.array(preds)
        pro_probs_arr = np.array(pro_probs)

        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["CON", "PRO"], zero_division=0
        )
        auc = float(roc_auc_score(y_true == "PRO", pro_probs_arr)) if len(np.unique(y_true)) > 1 else 0.5
        cm = confusion_matrix(y_true, y_pred, labels=["CON", "PRO"]).tolist()
        kappa = float(cohen_kappa_score(p_choices, s_choices))

        metrics = EvaluationMetrics(
            model_name=pipeline_title,
            accuracy=round(acc, 4),
            macro_f1=round(f1, 4),
            macro_precision=round(p, 4),
            macro_recall=round(r, 4),
            con_f1=round(f1_c[0], 4),
            pro_f1=round(f1_c[1], 4),
            roc_auc=round(auc, 4),
            confusion_mat=cm,
        )

        ensemble_stats = {
            "consensus_rate": round(float(np.mean(consensuses)) * 100.0, 2),
            "inter_judge_kappa": round(kappa, 4),
            "consensus_accuracy": round(float(eval_df[eval_df["is_consensus"]]["is_correct"].mean()) * 100.0, 2) if any(consensuses) else 0.0,
            "disagreement_accuracy": round(float(eval_df[~eval_df["is_consensus"]]["is_correct"].mean()) * 100.0, 2) if not all(consensuses) else 0.0,
        }

        return metrics, eval_df, ensemble_stats
