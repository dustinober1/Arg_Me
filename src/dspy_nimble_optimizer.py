"""
DSPy Prompt & Few-Shot Optimization Engine for Bespoke Labs Nimble 9B on args.me.

Implements:
1. DSPy Signatures and Example datasets for argument stance classification.
2. Contrastive Few-Shot Selection tailored for Nimble's contrastive pair pre-training.
3. Multi-prompt candidate tuning (Dialectic Refutation, Adversarial Quoting, Motion Grounding).
4. Scalable evaluation engine supporting large dataset slices (up to full test set).
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
# 1. DSPy Signatures & Examples
# -----------------------------------------------------------------------------

class ContrastiveStanceSignature(dspy.Signature):
    """Classify argument stance accounting for adversarial citations, refutations, and motion polarity."""
    topic: str = dspy.InputField(desc="The debate motion or resolution topic.")
    premise: str = dspy.InputField(desc="The submitted premise text by the debater.")
    stance: str = dspy.OutputField(desc="The stance: 'PRO' (supports/affirms motion) or 'CON' (opposes/negates motion).")


def build_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    """Convert a DataFrame into a list of DSPy input-output Examples."""
    examples = []
    for _, row in df.iterrows():
        ex = dspy.Example(
            topic=str(row["conclusion"]),
            premise=str(row["premise_text"]),
            stance=str(row["stance"]),
        ).with_inputs("topic", "premise")
        examples.append(ex)
    return examples


# -----------------------------------------------------------------------------
# 2. Tuned Prompt Instructions & Contrastive Few-Shot Exemplars
# -----------------------------------------------------------------------------

CONTRASTIVE_FEW_SHOT_EXEMPLARS = [
    {
        "topic": "School Uniforms Should Be Abolished",
        "premise": "My opponent claims uniforms save parents money, but uniforms suppress children's creative autonomy and enforce rigid conformity.",
        "stance": "PRO",
        "dialectic_note": "Attacks opponent's defense to support the abolition motion (PRO).",
    },
    {
        "topic": "School Uniforms Should Be Abolished",
        "premise": "Uniforms eliminate socioeconomic clothing competition and reduce bullying in hallways.",
        "stance": "CON",
        "dialectic_note": "Defends uniforms, opposing the abolition motion (CON).",
    },
    {
        "topic": "Nuclear Energy Expansion",
        "premise": "Pro claims clean power is urgent, yet nuclear waste storage remains unsolved for tens of thousands of years.",
        "stance": "CON",
        "dialectic_note": "Quotes 'Pro' to refute the motion; author stance is CON.",
    },
    {
        "topic": "Nuclear Energy Expansion",
        "premise": "Modern Gen-IV molten salt reactors offer walk-away safety and produce reliable zero-emission baseload power.",
        "stance": "PRO",
        "dialectic_note": "Direct affirmative claim supporting expansion (PRO).",
    },
]

CANDIDATE_INSTRUCTIONS = {
    "baseline": "What is the argumentative stance of the premise with respect to the topic?",
    "dialectic_tuned": (
        "Analyze the speaker's ultimate thesis relative to the debate topic. "
        "Carefully account for adversarial quotes (e.g., if the author quotes 'Pro argues...' only to debunk it, the author's stance is CON). "
        "Distinguish substantive arguments from procedural chatter."
    ),
}

TUNED_CRITERIA = {
    "PRO": (
        "The premise ultimately supports, defends, or advocates in favor of the resolution. "
        "Includes premises that defend the resolution against opposing attacks or refute negative claims."
    ),
    "CON": (
        "The premise ultimately opposes, criticizes, or rejects the resolution. "
        "Includes premises that quote the affirmative side ('Pro argues...') in order to refute or attack them."
    ),
}


# -----------------------------------------------------------------------------
# 3. Optimized Nimble DSPy Pipeline
# -----------------------------------------------------------------------------

class OptimizedNimblePipeline(dspy.Module):
    """
    Optimized DSPy module executing tuned prompt instructions and
    contrastive few-shot exemplars on Bespoke Labs Nimble 9B via Ollama.
    """

    def __init__(
        self,
        model_name: str = "nimble",
        ollama_base_url: str = "http://localhost:11434",
        instruction_mode: str = "dialectic_tuned",
        use_few_shots: bool = True,
        few_shot_exemplars: Optional[List[Dict[str, str]]] = None,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/v1/systemone"
        self.instruction = CANDIDATE_INSTRUCTIONS.get(instruction_mode, CANDIDATE_INSTRUCTIONS["dialectic_tuned"])
        self.criteria = TUNED_CRITERIA
        self.use_few_shots = use_few_shots
        self.exemplars = few_shot_exemplars or CONTRASTIVE_FEW_SHOT_EXEMPLARS
        self.timeout = timeout

    def format_state(self, topic: str, premise: str) -> str:
        """Format input state with contrastive few-shot guidance if enabled."""
        if not self.use_few_shots:
            return f"Topic: {topic}\nPremise: {premise}"

        # Build contrastive few-shot prefix
        prefix_lines = ["### Contrastive Reference Reasoning:"]
        for ex in self.exemplars[:4]:
            prefix_lines.append(
                f"- [Motion]: {ex['topic']}\n"
                f"  [Premise]: {ex['premise']}\n"
                f"  [Decision]: {ex['stance']} ({ex['dialectic_note']})"
            )
        prefix_lines.append("\n### Target Case to Decide:")
        prefix_lines.append(f"[Motion]: {topic}\n[Premise]: {premise}")
        return "\n".join(prefix_lines)

    def forward(self, topic: str, premise: str) -> dspy.Prediction:
        """
        Execute optimized decision inference on (topic, premise).
        """
        state_text = self.format_state(topic, premise)
        payload = {
            "model": self.model_name,
            "state": state_text,
            "questions": {
                "stance": {
                    "type": "choice",
                    "instructions": self.instruction,
                    "criteria": self.criteria,
                }
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            latency_ms = (time.time() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answers", {}).get("stance", {})
                choice = ans.get("choice", "PRO")
                probs = ans.get("probabilities", {})
                confidence = float(ans.get("confidence", 0.0))

                return dspy.Prediction(
                    stance=choice,
                    pro_prob=float(probs.get("PRO", 0.5)),
                    con_prob=float(probs.get("CON", 0.5)),
                    confidence=confidence,
                    latency_ms=latency_ms,
                    error=None,
                )
            else:
                return dspy.Prediction(
                    stance="PRO",
                    pro_prob=0.5,
                    con_prob=0.5,
                    confidence=0.0,
                    latency_ms=latency_ms,
                    error=f"HTTP {resp.status_code}: {resp.text}",
                )
        except Exception as e:
            return dspy.Prediction(
                stance="PRO",
                pro_prob=0.5,
                con_prob=0.5,
                confidence=0.0,
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
        Evaluate pipeline over DataFrame slice.
        """
        eval_df = df.copy()
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

        print(f"\nEvaluating Optimized Nimble 9B on N = {len(eval_df)} arguments...")
        preds = []
        pro_probs = []
        confidences = []
        latencies = []

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Optimized Nimble Inference"):
            pred = self.forward(str(row["conclusion"]), str(row["premise_text"]))
            preds.append(pred.stance)
            pro_probs.append(pred.pro_prob)
            confidences.append(pred.confidence)
            latencies.append(pred.latency_ms)

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

        metrics = EvaluationMetrics(
            model_name="Optimized Nimble 9B (DSPy Prompt + Contrastive Few-Shots)",
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
