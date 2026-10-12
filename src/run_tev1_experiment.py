"""
Runner for evaluating Together AI Tev1:4B Decision Model on args.me debate arguments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR
from src.ollama_decision_model import Tev1DecisionClassifier


def run_tev1_benchmark(sample_size: int = 250):
    parquet_path = DATA_DIR / "args_me_benchmark_5k.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {parquet_path}")

    print(f"Loading benchmark dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df_clean = df[df["stance"].isin(["PRO", "CON"])].copy().reset_index(drop=True)

    # Use exact same test split as the traditional & transformer benchmarks
    _, test_df = train_test_split(
        df_clean, test_size=0.20, random_state=42, stratify=df_clean["stance"]
    )

    print(f"Held-out test set size: {len(test_df)}")
    print(f"Evaluating a representative sample of N = {sample_size} test arguments...")

    classifier = Tev1DecisionClassifier(model_name="tev1:4b")
    metrics, results_df = classifier.evaluate_df(test_df, sample_size=sample_size, random_state=42)

    print("\n" + "=" * 80)
    print("TEV1:4B DECISION MODEL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model:           {metrics.model_name}")
    print(f"Accuracy:        {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    print(f"Macro F1:        {metrics.macro_f1:.4f}")
    print(f"Macro Precision: {metrics.macro_precision:.4f}")
    print(f"Macro Recall:    {metrics.macro_recall:.4f}")
    print(f"CON F1:          {metrics.con_f1:.4f} | PRO F1: {metrics.pro_f1:.4f}")
    print(f"ROC-AUC:         {metrics.roc_auc:.4f}")
    print(f"Confusion Matrix (CON, PRO): {metrics.confusion_mat}")
    print(f"Average Latency per Decision: {results_df['latency_ms'].mean():.1f} ms")

    # Paradigm comparison
    print("\n" + "=" * 80)
    print("CROSS-PARADIGM BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Paradigm / Model':<45} | {'Accuracy':<8} | {'Macro F1':<8} | {'ROC-AUC':<8}")
    print("-" * 75)
    print(f"{'Majority Baseline (Zero-R)':<45} | 0.5020   | 0.3342   | 0.5000")
    print(f"{'VADER Sentiment Polarity':<45} | 0.5040   | 0.4988   | N/A")
    print(f"{'Linear SVM (Contextual TF-IDF)':<45} | 0.5070   | 0.5067   | 0.5244")
    print(f"{'Logistic Regression (Premise TF-IDF)':<45} | 0.5490   | 0.5490   | 0.5580")
    print(f"{'Zero-Shot NLI DeBERTa-v3 Small':<45} | 0.5230   | 0.4936   | 0.5541")
    print(f"{'Fine-Tuned RoBERTa-base Cross-Encoder':<45} | 0.5890   | 0.5868   | 0.6265")
    print(f"{metrics.model_name:<45} | {metrics.accuracy:<8.4f} | {metrics.macro_f1:<8.4f} | {metrics.roc_auc:<8.4f}")

    # Inspect sample errors
    print("\n" + "=" * 80)
    print("SAMPLE ERROR ANALYSIS (WHERE TEV1 DISAGREED WITH GROUND TRUTH)")
    print("=" * 80)
    errors_df = results_df[~results_df["is_correct"]].head(5)
    for idx, row in errors_df.iterrows():
        print(f"\n[Topic]: {row['conclusion']}")
        print(f"[Premise Snippet]: {row['premise_text'][:200]}...")
        print(f"[Ground Truth]: {row['stance']} | [Tev1 Prediction]: {row['pred_stance']} (PRO Prob: {row['pro_prob']:.3f}, Conf: {row['confidence']:.3f})")

    return metrics, results_df


if __name__ == "__main__":
    run_tev1_benchmark(sample_size=200)
