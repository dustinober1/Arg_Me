"""
Client and Benchmark Runner for Together AI Tev1 Decision Models via Ollama.

Tev1 is a family of decision models fine-tuned from Qwen (4B / 0.8B) for fast classification.
Instead of autoregressive free-form text generation, decision models evaluate options
directly via Ollama's `/v1/systemone` endpoint, returning exact probability distributions
and chosen options in zero generation tokens.
"""

import time
import json
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
)

from src.baseline_stance import EvaluationMetrics


class Tev1DecisionClassifier:
    """
    Evaluates stance classification using Together AI's Tev1 decision model on Ollama.
    """

    DEFAULT_CRITERIA = {
        "PRO": "The premise supports, defends, or argues in favor of the motion/topic.",
        "CON": "The premise opposes, attacks, refutes, or argues against the motion/topic.",
    }

    def __init__(
        self,
        model_name: str = "tev1:4b",
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/v1/systemone"
        self.timeout = timeout
        self.verify_model_availability()

    def verify_model_availability(self) -> bool:
        """Check if Ollama is running and the model is loaded/available."""
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                tags = resp.json().get("models", [])
                names = [m.get("name", "") for m in tags]
                # Check if model name or prefix matches
                matching = any(self.model_name in name or name.startswith("tev1") for name in names)
                if not matching:
                    print(f"Warning: Model '{self.model_name}' was not found in active Ollama tags: {names}")
                else:
                    print(f"Successfully verified '{self.model_name}' is available on Ollama.")
                return True
        except Exception as e:
            print(f"Warning: Could not connect to Ollama at {self.ollama_base_url}: {e}")
            return False
        return True

    def predict_single(
        self,
        conclusion: str,
        premise_text: str,
    ) -> Dict[str, Any]:
        """
        Send a single argument pair to the /v1/systemone decision endpoint.
        """
        payload = {
            "model": self.model_name,
            "state": f"Topic: {conclusion}\nPremise: {premise_text}",
            "questions": {
                "stance": {
                    "type": "choice",
                    "instructions": "What is the argumentative stance of the premise with respect to the topic?",
                    "criteria": self.DEFAULT_CRITERIA,
                }
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            elapsed_ms = (time.time() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                stance_ans = data.get("answers", {}).get("stance", {})
                choice = stance_ans.get("choice", "UNKNOWN")
                probs = stance_ans.get("probabilities", {})
                confidence = stance_ans.get("confidence", 0.0)

                return {
                    "choice": choice,
                    "pro_prob": float(probs.get("PRO", 0.5)),
                    "con_prob": float(probs.get("CON", 0.5)),
                    "confidence": float(confidence),
                    "latency_ms": elapsed_ms,
                    "error": None,
                }
            else:
                return {
                    "choice": "UNKNOWN",
                    "pro_prob": 0.5,
                    "con_prob": 0.5,
                    "confidence": 0.0,
                    "latency_ms": elapsed_ms,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            return {
                "choice": "UNKNOWN",
                "pro_prob": 0.5,
                "con_prob": 0.5,
                "confidence": 0.0,
                "latency_ms": (time.time() - t0) * 1000.0,
                "error": str(e),
            }

    def evaluate_df(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[EvaluationMetrics, pd.DataFrame]:
        """
        Evaluate Tev1 against a test DataFrame of debate arguments.
        """
        eval_df = df.copy()
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

        print(f"\nEvaluating '{self.model_name}' on {len(eval_df)} arguments via {self.endpoint}...")
        
        preds = []
        pro_probs = []
        latencies = []
        confidences = []
        errors = 0

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Tev1 Decision Inference"):
            c = str(row.get("conclusion", ""))
            p = str(row.get("premise_text", ""))
            res = self.predict_single(c, p)

            if res["error"]:
                errors += 1
                # Default fallback
                preds.append("PRO")
                pro_probs.append(0.5)
            else:
                preds.append(res["choice"])
                pro_probs.append(res["pro_prob"])

            latencies.append(res["latency_ms"])
            confidences.append(res["confidence"])

        eval_df["pred_stance"] = preds
        eval_df["pro_prob"] = pro_probs
        eval_df["confidence"] = confidences
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
        print(f"\nInference completed. Mean latency per decision: {avg_latency:.1f} ms | Errors: {errors}")

        metrics = EvaluationMetrics(
            model_name=f"Tev1 4B Decision Model ({self.model_name})",
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
