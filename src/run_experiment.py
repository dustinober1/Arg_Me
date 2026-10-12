"""
CLI Runner for Rule-Based & Traditional NLP Stance Prediction Experiments on args.me.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import DATA_DIR
from src.baseline_stance import StanceBenchmarkRunner
from src.dataset_loader import load_sample_df


def run_stance_benchmark():
    parquet_path = DATA_DIR / "args_me_benchmark_5k.parquet"
    if parquet_path.exists():
        print(f"Loading existing benchmark dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
    else:
        print("Benchmark dataset not found locally. Streaming 5,000 records...")
        df = load_sample_df(n_records=5000, save_parquet=True, parquet_filename="args_me_benchmark_5k.parquet")

    print(f"Total premises in evaluation set: {len(df)}")
    print(f"Stance class counts:\n{df['stance'].value_counts()}\n")

    runner = StanceBenchmarkRunner(random_state=42)

    print("================================================================================")
    print("RUNNING TRADITIONAL NLP & RULE-BASED STANCE DETECTION BENCHMARK (N = 5,000)")
    print("================================================================================")
    results_df = runner.run_comprehensive_benchmark(df, test_size=0.20)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY (Test Set N = 1,000)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP LEXICAL PREDICTORS (LOGISTIC REGRESSION COEFFICIENTS)")
    print("=" * 80)
    top_pro, top_con = runner.extract_top_features(df, n_top=15)

    print("\nTop 15 Indicative PRO Features (Positive Weights):")
    for word, weight in top_pro:
        print(f"  + {word:<25} (coef = {weight:+.4f})")

    print("\nTop 15 Indicative CON Features (Negative Weights):")
    for word, weight in top_con:
        print(f"  - {word:<25} (coef = {weight:+.4f})")

    print("\n" + "=" * 80)
    print("CROSS-TOPIC GENERALIZATION TEST (EVALUATING ON UNSEEN DEBATE TOPICS)")
    print("=" * 80)
    oot_metrics = runner.run_cross_topic_generalization(df)
    print(f"Model: {oot_metrics.model_name}")
    print(f"Accuracy:         {oot_metrics.accuracy:.4f}")
    print(f"Macro F1:         {oot_metrics.macro_f1:.4f}")
    print(f"Macro Precision:  {oot_metrics.macro_precision:.4f}")
    print(f"Macro Recall:     {oot_metrics.macro_recall:.4f}")
    print(f"CON F1:           {oot_metrics.con_f1:.4f} | PRO F1: {oot_metrics.pro_f1:.4f}")

    return results_df, top_pro, top_con, oot_metrics


if __name__ == "__main__":
    run_stance_benchmark()
