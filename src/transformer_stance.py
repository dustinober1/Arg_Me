"""
Transformer Cross-Encoder Architecture for Debate Stance Detection on args.me.

Implements:
1. Paired sequence tokenization [CLS] Conclusion [SEP] Premise [SEP].
2. Cross-Encoder fine-tuning with DeBERTa-v3 / RoBERTa backbones.
3. Zero-shot NLI entailment-based stance classification baseline.
4. Hardware-aware device orchestration (CUDA / Apple Silicon MPS / CPU).
5. Comprehensive metric evaluation matching the benchmark framework.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from src.baseline_stance import EvaluationMetrics


# -----------------------------------------------------------------------------
# 1. Device Orchestration
# -----------------------------------------------------------------------------

def get_compute_device() -> torch.device:
    """Detect and return the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -----------------------------------------------------------------------------
# 2. PyTorch Dataset for Paired Cross-Encoding
# -----------------------------------------------------------------------------

class DebateArgumentDataset(Dataset):
    """
    Dataset for paired sequence classification.
    Input structure: [CLS] conclusion [SEP] premise [SEP]
    """

    LABEL_MAP = {"CON": 0, "PRO": 1}
    INV_LABEL_MAP = {0: "CON", 1: "PRO"}

    def __init__(
        self,
        conclusions: List[str],
        premises: List[str],
        labels: Optional[List[str]] = None,
        tokenizer: Any = None,
        max_length: int = 256,
    ):
        self.conclusions = [str(c) for c in conclusions]
        self.premises = [str(p) for p in premises]
        self.labels = [self.LABEL_MAP[l] for l in labels] if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conclusions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        c_text = self.conclusions[idx]
        p_text = self.premises[idx]

        encoding = self.tokenizer(
            c_text,
            p_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# -----------------------------------------------------------------------------
# 3. Transformer Cross-Encoder Classifier
# -----------------------------------------------------------------------------

class TransformerStanceCrossEncoder:
    """
    Fine-tunes and evaluates a Transformer Cross-Encoder for stance classification.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_labels: int = 2,
        max_length: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or get_compute_device()

        print(f"Initializing Cross-Encoder with backbone: '{model_name}' on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={0: "CON", 1: "PRO"},
            label2id={"CON": 0, "PRO": 1},
        )
        self.model.to(self.device)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, float]]:
        """
        Train the cross-encoder on premise-conclusion pairs.
        """
        train_dataset = DebateArgumentDataset(
            conclusions=train_df["conclusion"].tolist(),
            premises=train_df["premise_text"].tolist(),
            labels=train_df["stance"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_df is not None:
            val_dataset = DebateArgumentDataset(
                conclusions=val_df["conclusion"].tolist(),
                premises=val_df["premise_text"].tolist(),
                labels=val_df["stance"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = []
        best_val_f1 = -1.0
        best_weights_path = Path("best_cross_encoder_weights.pt")

        print(f"\nStarting training for {epochs} epochs ({total_steps} total optimization steps)...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**model_inputs)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            epoch_log = {"epoch": epoch, "train_loss": avg_train_loss}

            if val_loader is not None:
                val_metrics = self.evaluate_loader(val_loader)
                epoch_log["val_loss"] = val_metrics["loss"]
                epoch_log["val_acc"] = val_metrics["accuracy"]
                epoch_log["val_macro_f1"] = val_metrics["macro_f1"]
                print(
                    f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
                )

                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    torch.save(self.model.state_dict(), best_weights_path)
            else:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}")

            history.append(epoch_log)

        # Reload best weights if validation was used
        if val_loader is not None and best_weights_path.exists():
            print(f"Reloading best model weights (Validation Macro F1 = {best_val_f1:.4f})...")
            self.model.load_state_dict(torch.load(best_weights_path, map_location=self.device, weights_only=True))
            if best_weights_path.exists():
                os.remove(best_weights_path)

        return history

    def predict(self, conclusions: List[str], premises: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and class probabilities for argument pairs.
        Returns:
            preds: Array of string labels ('CON', 'PRO')
            probs: Array of float probabilities for PRO class
        """
        dataset = DebateArgumentDataset(
            conclusions=conclusions,
            premises=premises,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**model_inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

        all_probs = np.array(all_probs)
        preds = np.where(all_probs >= 0.5, "PRO", "CON")
        return preds, all_probs

    def evaluate_loader(self, loader: DataLoader) -> Dict[str, float]:
        """Compute loss and metrics over an annotated DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**model_inputs)
                total_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": f1,
        }

    def evaluate_df(
        self,
        test_df: pd.DataFrame,
        benchmark_name: str = "Fine-Tuned DeBERTa-v3 Cross-Encoder",
    ) -> EvaluationMetrics:
        """
        Evaluate against a test DataFrame and return an EvaluationMetrics object.
        """
        y_true = test_df["stance"].values
        preds, probs = self.predict(
            conclusions=test_df["conclusion"].tolist(),
            premises=test_df["premise_text"].tolist(),
        )

        acc = float(accuracy_score(y_true, preds))
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true, preds, labels=["CON", "PRO"], zero_division=0
        )
        auc = float(roc_auc_score(y_true == "PRO", probs))
        cm = confusion_matrix(y_true, preds, labels=["CON", "PRO"]).tolist()

        return EvaluationMetrics(
            model_name=benchmark_name,
            accuracy=round(acc, 4),
            macro_f1=round(f1, 4),
            macro_precision=round(p, 4),
            macro_recall=round(r, 4),
            con_f1=round(f1_c[0], 4),
            pro_f1=round(f1_c[1], 4),
            roc_auc=round(auc, 4),
            confusion_mat=cm,
        )


# -----------------------------------------------------------------------------
# 4. Zero-Shot Pre-Trained NLI Stance Classifier
# -----------------------------------------------------------------------------

class ZeroShotNLIStanceClassifier:
    """
    Evaluates stance without task-specific fine-tuning by mapping Natural Language
    Inference (NLI) relationship between Conclusion (premise) and Premise (hypothesis):
    - Entailment -> PRO
    - Contradiction -> CON
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[torch.device] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device or get_compute_device()
        self.batch_size = batch_size
        print(f"Loading Zero-Shot NLI Cross-Encoder: '{model_name}' on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, conclusions: List[str], premises: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        NLI models predict: 0: contradiction, 1: entailment, 2: neutral (or similar).
        We map relative probabilities of Entailment vs Contradiction to PRO vs CON.
        """
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        entail_idx = next(i for i, l in id2label.items() if "entail" in l)
        contra_idx = next(i for i, l in id2label.items() if "contradict" in l)

        dataset = DebateArgumentDataset(
            conclusions=conclusions,
            premises=premises,
            tokenizer=self.tokenizer,
            max_length=256,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        pro_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)

                p_entail = probs[:, entail_idx]
                p_contra = probs[:, contra_idx]
                # Normalized relative probability between entailment and contradiction
                rel_pro = p_entail / (p_entail + p_contra + 1e-8)
                pro_probs.extend(rel_pro.cpu().numpy())

        pro_probs = np.array(pro_probs)
        preds = np.where(pro_probs >= 0.5, "PRO", "CON")
        return preds, pro_probs

    def evaluate_df(
        self,
        test_df: pd.DataFrame,
        benchmark_name: str = "Zero-Shot NLI Cross-Encoder (Out-of-the-Box)",
    ) -> EvaluationMetrics:
        y_true = test_df["stance"].values
        preds, probs = self.predict(
            conclusions=test_df["conclusion"].tolist(),
            premises=test_df["premise_text"].tolist(),
        )

        acc = float(accuracy_score(y_true, preds))
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true, preds, labels=["CON", "PRO"], zero_division=0
        )
        auc = float(roc_auc_score(y_true == "PRO", probs))
        cm = confusion_matrix(y_true, preds, labels=["CON", "PRO"]).tolist()

        return EvaluationMetrics(
            model_name=benchmark_name,
            accuracy=round(acc, 4),
            macro_f1=round(f1, 4),
            macro_precision=round(p, 4),
            macro_recall=round(r, 4),
            con_f1=round(f1_c[0], 4),
            pro_f1=round(f1_c[1], 4),
            roc_auc=round(auc, 4),
            confusion_mat=cm,
        )
