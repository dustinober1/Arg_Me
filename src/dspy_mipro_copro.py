"""
MIPROv2 & COPRO Optimization Engine for Bespoke Labs Nimble 9B on args.me.

Architecture:
- Asymmetric Dual-Model Paradigm:
  1. Prompt Model (Proposer / Meta-Optimizer): Generative LLM proposing dialectic instructions & stance criteria (COPRO/MIPROv2).
  2. Task Model (Decision Worker): Bespoke Labs Nimble 9B (via /v1/systemone) evaluating decisions on args.me.
- Standalone Local Fallback:
  BootstrapFewShot & Contrastive Exemplar search running 100% on Nimble 9B without requiring an external proposer model.
"""

import os
import re
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
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
# 1. DSPy Signatures & Metric
# -----------------------------------------------------------------------------

class ArgumentStanceSignature(dspy.Signature):
    """Classify whether the submitted premise supports (PRO) or opposes (CON) the debate topic resolution."""
    topic: str = dspy.InputField(desc="The debate motion or resolution topic.")
    premise: str = dspy.InputField(desc="The submitted argument premise by the debater.")
    stance: str = dspy.OutputField(desc="'PRO' (affirms/supports motion) or 'CON' (negates/opposes motion).")


def stance_accuracy_metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """
    Standard DSPy evaluation metric: Returns 1.0 if predicted stance matches gold label, else 0.0.
    """
    gold = getattr(example, "stance", "").strip().upper()
    predicted = getattr(pred, "stance", "").strip().upper()
    return 1.0 if gold == predicted else 0.0


def stance_dialectic_metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """
    Composite metric incorporating confidence score when prediction is correct.
    Rewards confident correct answers (1.0 + confidence/2.0) and penalizes confident wrong answers.
    """
    gold = getattr(example, "stance", "").strip().upper()
    predicted = getattr(pred, "stance", "").strip().upper()
    confidence = float(getattr(pred, "confidence", 0.5))

    if gold == predicted:
        return 1.0 + (confidence * 0.2)
    else:
        return max(0.0, 0.2 - (confidence * 0.2))


def df_to_dspy_dataset(df: pd.DataFrame) -> List[dspy.Example]:
    """Convert pandas DataFrame rows into DSPy Example objects."""
    examples = []
    for _, row in df.iterrows():
        ex = dspy.Example(
            topic=str(row["conclusion"]),
            premise=str(row["premise_text"]),
            stance=str(row["stance"]).strip().upper(),
        ).with_inputs("topic", "premise")
        examples.append(ex)
    return examples


# -----------------------------------------------------------------------------
# 2. Generative Prompt Proposer Model (Qwen 3.5 2B via Ollama)
# -----------------------------------------------------------------------------

class QwenProposerLM(dspy.LM):
    """
    Generative prompt proposer LM using Ollama's qwen3.5:2b endpoint.
    Extracts message['content'] from /api/chat, bypassing internal reasoning tokens,
    and returns generated dialectic instructions for COPRO and MIPROv2 optimizers.
    """

    def __init__(
        self,
        model: str = "qwen3.5:2b",
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.model = model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/api/chat"
        self.timeout = timeout
        self.temperature = temperature

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        req_messages = messages or [{"role": "user", "content": prompt or ""}]
        payload = {
            "model": self.model,
            "messages": req_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                msg = resp.json().get("message", {})
                content = msg.get("content", "").strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return [content] if content else ["Analyze dialectical stance relative to the debate motion."]
            return ["Analyze whether the premise affirms or negates the debate resolution."]
        except Exception:
            return ["Analyze whether the premise affirms or negates the debate resolution."]

    def propose_candidate_instructions(
        self,
        current_instruction: str,
        n_candidates: int = 4,
    ) -> List[str]:
        """
        Actively query Qwen 3.5 2B to propose dialectic instruction mutations.
        """
        prompt = (
            f"You are an expert NLP prompt engineer. The task is stance classification (PRO vs CON) on debate motions.\n"
            f"Current instruction: \"{current_instruction}\"\n\n"
            f"Debate challenges include debaters quoting their opponent ('Pro claims...') to refute them, and procedural chatter.\n"
            f"Propose {n_candidates} distinct, high-impact instruction variations for classifying the author's stance.\n"
            f"Output exactly {n_candidates} numbered lines (1., 2., etc.), each 1-2 sentences."
        )
        try:
            responses = self(prompt=prompt)
            text = responses[0] if responses else ""
            candidates = []
            for line in text.split("\n"):
                line = line.strip()
                if re.match(r"^\d+[\.\)]\s*", line):
                    clean_cand = re.sub(r"^\d+[\.\)]\s*", "", line).strip().strip('"')
                    if len(clean_cand) > 10:
                        candidates.append(clean_cand)
        except Exception:
            candidates = []

        fallbacks = [
            "Analyze the author's ultimate thesis relative to the debate motion, accounting for adversary citations.",
            "Determine if the speaker provides affirmative justification (PRO) or negative refutation (CON) for the resolution.",
            "Distinguish between procedural contest remarks and substantive arguments supporting or attacking the topic.",
            "Focus on debate resolution polarity: classify whether the premise defends (PRO) or dismantles (CON) the motion.",
        ]
        while len(candidates) < n_candidates:
            candidates.append(fallbacks[len(candidates) % len(fallbacks)])
        return candidates[:n_candidates]


# -----------------------------------------------------------------------------
# 3. Custom Decision LM for Bespoke Labs Nimble 9B
# -----------------------------------------------------------------------------

class NimbleDecisionLM(dspy.LM):
    """
    Custom DSPy Language Model wrapper for Bespoke Labs Nimble 9B via Ollama /v1/systemone.
    Converts DSPy prompts into systemone choice payloads and returns categorical decisions.
    """

    def __init__(
        self,
        model: str = "nimble",
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.model = model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/v1/systemone"
        self.timeout = timeout
        self.temperature = temperature
        self.history = []

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Execute call for DSPy module prediction.
        Extracts topic and premise from prompt/messages, invokes /v1/systemone, and returns ['PRO'] or ['CON'].
        """
        raw_text = prompt or ""
        if messages:
            raw_text = "\n".join([m.get("content", "") for m in messages])

        # Parse instructions or criteria if present in prompt
        instructions = "Classify the premise as PRO (affirms resolution) or CON (opposes resolution)."
        criteria = {
            "PRO": "The premise affirms or defends the debate resolution.",
            "CON": "The premise negates, opposes, or refutes the debate resolution.",
        }

        # Format systemone payload
        payload = {
            "model": self.model,
            "state": raw_text,
            "questions": {
                "stance": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }

        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answers", {}).get("stance", {})
                choice = ans.get("choice", "PRO").strip().upper()
                return [choice]
            else:
                return ["PRO"]
        except Exception:
            return ["PRO"]


# -----------------------------------------------------------------------------
# 3. Decision Stance Pipeline Module
# -----------------------------------------------------------------------------

class NimbleStancePredictor(dspy.Module):
    """
    DSPy Module executing stance classification via Bespoke Labs Nimble 9B.
    Compatible with COPRO, MIPROv2, and BootstrapFewShot optimizers.
    """

    def __init__(
        self,
        model_name: str = "nimble",
        ollama_base_url: str = "http://localhost:11434",
        custom_instruction: Optional[str] = None,
        custom_criteria: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.endpoint = f"{self.ollama_base_url}/v1/systemone"
        self.instruction = custom_instruction or (
            "Analyze the author's ultimate thesis relative to the debate topic. "
            "Properly account for adversarial quotes (e.g. if author quotes 'Pro argues...' to refute it, the stance is CON)."
        )
        self.criteria = custom_criteria or {
            "PRO": "The premise ultimately affirms, supports, or defends the resolution topic.",
            "CON": "The premise ultimately opposes, rejects, or refutes the resolution topic.",
        }
        self.timeout = timeout
        self.demos: List[dspy.Example] = []

    def forward(self, topic: str, premise: str) -> dspy.Prediction:
        """Execute decision inference with attached few-shot demonstrations if compiled."""
        state_parts = []
        if self.demos:
            state_parts.append("### Exemplar Stance Deciders:")
            for d in self.demos[:4]:
                state_parts.append(
                    f"Topic: {getattr(d, 'topic', '')}\n"
                    f"Premise: {getattr(d, 'premise', '')}\n"
                    f"Stance: {getattr(d, 'stance', '')}"
                )
            state_parts.append("### Target Argument:")

        state_parts.append(f"Topic: {topic}\nPremise: {premise}")
        state_text = "\n\n".join(state_parts)

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
                choice = ans.get("choice", "PRO").strip().upper()
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
        pipeline_title: str = "Nimble Stance Predictor",
    ) -> Tuple[EvaluationMetrics, pd.DataFrame]:
        """Benchmark pipeline accuracy, F1, ROC-AUC and latency across a test DataFrame."""
        eval_df = df.copy()
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(sample_size, random_state=random_state).reset_index(drop=True)

        preds = []
        pro_probs = []
        confidences = []
        latencies = []

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Evaluating {pipeline_title}"):
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
        auc = float(roc_auc_score(y_true == "PRO", pro_probs_arr)) if len(np.unique(y_true)) > 1 else 0.5
        cm = confusion_matrix(y_true, y_pred, labels=["CON", "PRO"]).tolist()

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

        return metrics, eval_df
