"""
Advanced DSPy & Multi-Question Decision Pipeline for Argument Stance Classification.

Integrates Together AI's Tev1:4B Decision Model with DSPy paradigms:
1. Declarative Signatures (ArgumentStanceSignature, DialecticRubricSignature).
2. Multi-Question Joint Evaluation on Ollama /v1/systemone:
   - Evaluates 'stance', 'is_substantive' (filtering forum noise), and 'quotes_opponent' in a single forward pass.
3. Dialectic Adversary Adjustment:
   - Automatically disentangles opponent quotations (e.g. 'Pro argues that X...') from the author's own thesis.
4. DSPy-compatible Prediction Interface and Evaluation Metric.
"""

import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

import dspy
from src.baseline_stance import EvaluationMetrics


# -----------------------------------------------------------------------------
# 1. DSPy Declarative Signatures
# -----------------------------------------------------------------------------

class ArgumentStanceSignature(dspy.Signature):
    """Determine the argumentative stance of a premise with respect to the motion (conclusion)."""
    topic: str = dspy.InputField(desc="The central resolution, proposition, or debate motion.")
    premise: str = dspy.InputField(desc="The argumentative premise or body text submitted by the debater.")
    stance: str = dspy.OutputField(desc="The author's stance: 'PRO' (supports/affirms) or 'CON' (opposes/negates).")


class DialecticDecompositionSignature(dspy.Signature):
    """Jointly evaluate substantive quality, adversary quoting, and argumentative stance."""
    topic: str = dspy.InputField(desc="The debate motion.")
    premise: str = dspy.InputField(desc="The submitted premise text.")
    is_substantive: str = dspy.OutputField(desc="'substantive' if logical argument, 'chatter' if forum negotiation/forfeit.")
    quotes_opponent: bool = dspy.OutputField(desc="True if the author quotes or references their adversary's claim.")
    stance: str = dspy.OutputField(desc="The ultimate stance: PRO or CON.")


# -----------------------------------------------------------------------------
# 2. Advanced Multi-Question Decision Engine (Tev1 on /v1/systemone)
# -----------------------------------------------------------------------------

class Tev1MultiQuestionPipeline(dspy.Module):
    """
    Advanced Decision Pipeline executing joint multi-question rubrics
    via Ollama's /v1/systemone endpoint.
    """

    ENHANCED_QUESTIONS = {
        "is_substantive": {
            "type": "choice",
            "instructions": "Is this text a substantive argumentative claim with reasoning/evidence, or is it merely procedural debate chatter, turn negotiation, a single-word extension, or a forfeit?",
            "criteria": {
                "substantive": "Presents logical reasons, factual evidence, or rhetorical arguments addressing the motion.",
                "chatter": "Procedural chatter, turn negotiation, forfeit, single-word response (e.g. 'Extend'), or meta-commentary.",
            },
        },
        "quotes_opponent": {
            "type": "noul",
            "instructions": "Does the speaker quote, reference, or attack arguments made by their opponent (e.g. 'Pro claims', 'my opponent states')?",
        },
        "stance": {
            "type": "choice",
            "instructions": "Accounting for any refutations of opponent quotes, what is the ultimate stance of the speaker regarding the topic?",
            "criteria": {
                "PRO": "The author ultimately advocates in favor of the resolution/motion, or defends arguments supporting it against attacks.",
                "CON": "The author ultimately opposes, rejects, or attacks the resolution/motion, or defends the negative position against the affirmative.",
            },
        },
    }

    def __init__(
        self,
        model_name: str = "tev1:4b",
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/v1/systemone"
        self.timeout = timeout

    def forward(self, topic: str, premise: str) -> dspy.Prediction:
        """
        Execute multi-question joint inference on (topic, premise).
        Returns a dspy.Prediction object with stance, probabilities, and dialectic diagnostics.
        """
        payload = {
            "model": self.model_name,
            "state": f"Topic: {topic}\nPremise: {premise}",
            "questions": self.ENHANCED_QUESTIONS,
        }

        t0 = time.time()
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            elapsed_ms = (time.time() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                answers = data.get("answers", {})

                # 1. Substantive assessment
                sub_ans = answers.get("is_substantive", {})
                is_sub = sub_ans.get("choice", "substantive")
                sub_probs = sub_ans.get("probabilities", {})

                # 2. Opponent quotation check
                quote_ans = answers.get("quotes_opponent", {})
                p_quote = float(quote_ans.get("noul", 0.0))

                # 3. Core Stance decision
                stance_ans = answers.get("stance", {})
                choice = stance_ans.get("choice", "PRO")
                probs = stance_ans.get("probabilities", {})
                conf = float(stance_ans.get("confidence", 0.5))

                pro_p = float(probs.get("PRO", 0.5))
                con_p = float(probs.get("CON", 0.5))

                # Dialectic Adjustment Rule:
                # If debater declares "I am Con" or "I am Pro" in procedural chatter,
                # resolve explicit declaration
                premise_lower = premise.lower()
                final_choice = choice
                if "i am con" in premise_lower or "i am the con" in premise_lower:
                    final_choice = "CON"
                    con_p = max(con_p, 0.90)
                    pro_p = 1.0 - con_p
                elif "i am pro" in premise_lower or "i am the pro" in premise_lower:
                    final_choice = "PRO"
                    pro_p = max(pro_p, 0.90)
                    con_p = 1.0 - pro_p

                return dspy.Prediction(
                    stance=final_choice,
                    pro_prob=pro_p,
                    con_prob=con_p,
                    confidence=conf,
                    is_substantive=is_sub,
                    substantive_prob=float(sub_probs.get("substantive", 1.0)),
                    quotes_opponent=(p_quote >= 0.5),
                    quote_prob=p_quote,
                    latency_ms=elapsed_ms,
                    error=None,
                )
            else:
                return dspy.Prediction(
                    stance="PRO",
                    pro_prob=0.5,
                    con_prob=0.5,
                    confidence=0.0,
                    is_substantive="substantive",
                    quotes_opponent=False,
                    latency_ms=(time.time() - t0) * 1000.0,
                    error=f"HTTP {resp.status_code}: {resp.text}",
                )
        except Exception as e:
            return dspy.Prediction(
                stance="PRO",
                pro_prob=0.5,
                con_prob=0.5,
                confidence=0.0,
                is_substantive="substantive",
                quotes_opponent=False,
                latency_ms=(time.time() - t0) * 1000.0,
                error=str(e),
            )

    def evaluate_df(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[EvaluationMetrics, pd.DataFrame]:
        """
        Evaluate against a benchmark DataFrame.
        """
        eval_df = df.copy()
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

        print(f"\nEvaluating Advanced DSPy/Tev1 Pipeline on {len(eval_df)} arguments...")
        preds = []
        pro_probs = []
        latencies = []
        confidences = []
        is_sub_list = []
        quotes_list = []

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Advanced Multi-Question Pipeline"):
            pred = self.forward(str(row["conclusion"]), str(row["premise_text"]))
            preds.append(pred.stance)
            pro_probs.append(pred.pro_prob)
            latencies.append(pred.latency_ms)
            confidences.append(pred.confidence)
            is_sub_list.append(pred.is_substantive)
            quotes_list.append(pred.quotes_opponent)

        eval_df["pred_stance"] = preds
        eval_df["pro_prob"] = pro_probs
        eval_df["confidence"] = confidences
        eval_df["is_substantive"] = is_sub_list
        eval_df["quotes_opponent"] = quotes_list
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
        auc = float(roc_auc_score(y_true == "PRO", pro_probs_arr))
        cm = confusion_matrix(y_true, y_pred, labels=["CON", "PRO"]).tolist()

        avg_latency = float(np.mean(latencies))
        chatter_pct = (eval_df["is_substantive"] == "chatter").mean() * 100.0
        quote_pct = eval_df["quotes_opponent"].mean() * 100.0

        print(f"\nAdvanced Pipeline Results:")
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"Mean Latency: {avg_latency:.1f} ms | Chatter Detected: {chatter_pct:.1f}% | Opponent Quotes: {quote_pct:.1f}%")

        metrics = EvaluationMetrics(
            model_name="Advanced Multi-Question Tev1 + DSPy Pipeline",
            accuracy=round(acc, 4),
            macro_f1=round(f1, 4),
            macro_precision=round(p, 4),
            macro_recall=round(r, 4),
            con_f1=round(f1_c[0], 4),
            pro_f1=round(f1_c[1], 4),
            roc_auc=round(auc, 4),
            confusion_mat=cm,
        )

        return metrics, eval_df
