"""
Rule-Based and Traditional NLP Classifiers for Stance Detection on args.me.

Implements:
1. Rule-based heuristic classifiers (debate cues, lexical polarity, sentiment concordances).
2. Linguistic & stylistic feature engineering (VADER sentiment, modal verbs, rhetorical markers,
   pronoun distribution, premise-conclusion overlap).
3. Machine Learning models with TF-IDF representations (MultinomialNB, LogisticRegression, LinearSVC).
4. Hybrid models combining sparse n-gram features with dense linguistic features.
5. Cross-validation and out-of-topic generalization evaluations.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from scipy.sparse import hstack, csr_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# -----------------------------------------------------------------------------
# 1. Rule-Based Stance Classifier
# -----------------------------------------------------------------------------

class RuleBasedStanceClassifier(BaseEstimator, ClassifierMixin):
    """
    Heuristic rule-based stance detector combining:
    - Explicit debate stance declarations (e.g., 'I agree', 'pro side', 'I oppose').
    - Opponent concession/forfeit markers (e.g., 'opponent forfeited', 'con concedes').
    - Sentiment polarity relative to debate direction.
    - Negation concordance with conclusion keywords.
    """

    PRO_PATTERNS = [
        r"\bi\s+(?:strongly\s+)?(?:support|agree|affirm|concur|favor|endorse)\b",
        r"\b(?:in\s+favor\s+of|pro\s+side|affirmative\s+side|pro-choice|pro-life)\b",
        r"\b(?:i\s+propose|we\s+should|we\s+must|it\s+is\s+vital|it\s+is\s+essential)\b",
        r"\b(?:valid|beneficial|effective|positive\s+outcome|proven\s+right)\b",
        r"\b(?:con\s+concedes?|opponent\s+fails?|opponent\s+forfeited)\b",
    ]

    CON_PATTERNS = [
        r"\bi\s+(?:strongly\s+)?(?:oppose|disagree|refute|reject|dispute)\b",
        r"\b(?:against|negative\s+side|con\s+side|contrary\s+to)\b",
        r"\b(?:flawed|fallacy|false|untrue|harmful|detrimental|unacceptable)\b",
        r"\b(?:we\s+should\s+not|must\s+not|cannot\s+support|no\s+evidence)\b",
        r"\b(?:opponent\s+is\s+wrong|opponent's\s+claim\s+is\s+invalid)\b",
    ]

    def __init__(self, use_vader: bool = True, vader_threshold: float = 0.05):
        self.use_vader = use_vader
        self.vader_threshold = vader_threshold
        self._sia = SentimentIntensityAnalyzer() if use_vader else None
        self.classes_ = np.array(["CON", "PRO"])

    def fit(self, X: Any = None, y: Any = None):
        """Rule-based classifier requires no parameter fitting."""
        return self

    def _predict_single(self, premise: str, conclusion: str = "") -> str:
        premise_lower = str(premise).lower()
        
        # 1. Check explicit affirmative / opposition regex markers
        pro_score = 0
        con_score = 0
        for pattern in self.PRO_PATTERNS:
            if re.search(pattern, premise_lower):
                pro_score += 1.5
        for pattern in self.CON_PATTERNS:
            if re.search(pattern, premise_lower):
                con_score += 1.5

        # 2. VADER sentiment polarity
        if self._sia:
            scores = self._sia.polarity_scores(premise_lower)
            compound = scores["compound"]
            if compound > self.vader_threshold:
                pro_score += 1.0
            elif compound < -self.vader_threshold:
                con_score += 1.0

        # 3. Decision threshold (default to PRO if exact tie)
        if pro_score > con_score:
            return "PRO"
        elif con_score > pro_score:
            return "CON"
        else:
            return "PRO"

    def predict(self, X: List[str] | pd.Series | np.ndarray) -> np.ndarray:
        preds = [self._predict_single(text) for text in X]
        return np.array(preds)


# -----------------------------------------------------------------------------
# 2. Dense Linguistic Feature Extractor
# -----------------------------------------------------------------------------

class DenseFeatureExtractor:
    """
    Extracts interpretable linguistic, stylistic, and overlap features
    from (premise, conclusion) pairs for traditional ML models.
    """

    MODAL_VERBS = {"should", "must", "ought", "would", "could", "shall", "cannot"}
    NEGATION_WORDS = {"not", "no", "never", "none", "neither", "nor", "hardly", "barely", "scarcely", "without"}
    FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    SECOND_THIRD_PERSON = {"you", "your", "yours", "they", "them", "their", "opponent", "opponents"}

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def extract_features_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical feature matrix from a DataFrame containing
        'premise_text' and 'conclusion' columns.
        """
        features_list = []
        for _, row in df.iterrows():
            premise = str(row.get("premise_text", "")).lower()
            conclusion = str(row.get("conclusion", "")).lower()

            # Word tokens
            p_tokens = re.findall(r"\b\w+\b", premise)
            c_tokens = set(re.findall(r"\b\w+\b", conclusion))
            p_len = max(len(p_tokens), 1)

            # Sentiment features
            s_scores = self.sia.polarity_scores(premise)

            # Token overlap with conclusion
            overlap_count = sum(1 for token in p_tokens if token in c_tokens)
            overlap_ratio = overlap_count / p_len
            jaccard = (
                len(set(p_tokens) & c_tokens) / len(set(p_tokens) | c_tokens)
                if (set(p_tokens) | c_tokens)
                else 0.0
            )

            # Linguistic & Stylistic markers
            neg_count = sum(1 for token in p_tokens if token in self.NEGATION_WORDS)
            modal_count = sum(1 for token in p_tokens if token in self.MODAL_VERBS)
            first_p_count = sum(1 for token in p_tokens if token in self.FIRST_PERSON)
            second_p_count = sum(1 for token in p_tokens if token in self.SECOND_THIRD_PERSON)

            # Punctuation & Rhetorical counts
            qmark_count = premise.count("?")
            exclam_count = premise.count("!")
            quote_count = premise.count('"') + premise.count("'")

            features_list.append([
                s_scores["compound"],
                s_scores["pos"],
                s_scores["neg"],
                s_scores["neu"],
                overlap_ratio,
                jaccard,
                neg_count / p_len,
                modal_count / p_len,
                first_p_count / p_len,
                second_p_count / p_len,
                qmark_count / p_len,
                exclam_count / p_len,
                quote_count / p_len,
                float(len(premise)),
                float(p_len),
                np.log1p(p_len),
            ])

        return np.array(features_list, dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "vader_compound",
            "vader_pos",
            "vader_neg",
            "vader_neu",
            "conclusion_overlap_ratio",
            "conclusion_jaccard_similarity",
            "negation_density",
            "modal_verb_density",
            "first_person_density",
            "second_third_person_density",
            "question_mark_density",
            "exclamation_mark_density",
            "quote_density",
            "char_length",
            "word_count",
            "log_word_count",
        ]


# -----------------------------------------------------------------------------
# 3. Model Benchmark Engine
# -----------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    model_name: str
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    pro_f1: float
    con_f1: float
    roc_auc: Optional[float] = None
    confusion_mat: Optional[List[List[int]]] = None


class StanceBenchmarkRunner:
    """
    Orchestrates standardized model training, cross-validation, and
    metric evaluation for debate stance classification.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results: List[EvaluationMetrics] = []
        self.dense_extractor = DenseFeatureExtractor()

    def evaluate_model(
        self,
        name: str,
        model: Any,
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
        is_rule_based: bool = False,
    ) -> EvaluationMetrics:
        """Fit model (if applicable) and evaluate against test split."""
        if not is_rule_based:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        
        # Calculate probabilities or decision scores for ROC-AUC if supported
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test == "PRO", probs))
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_test)
                auc = float(roc_auc_score(y_test == "PRO", scores))
            except Exception:
                pass

        acc = float(accuracy_score(y_test, preds))
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
            y_test, preds, labels=["CON", "PRO"], zero_division=0
        )
        cm = confusion_matrix(y_test, preds, labels=["CON", "PRO"]).tolist()

        metrics = EvaluationMetrics(
            model_name=name,
            accuracy=round(acc, 4),
            macro_f1=round(f1, 4),
            macro_precision=round(p, 4),
            macro_recall=round(r, 4),
            con_f1=round(f1_cls[0], 4),
            pro_f1=round(f1_cls[1], 4),
            roc_auc=round(auc, 4) if auc is not None else None,
            confusion_mat=cm,
        )
        self.results.append(metrics)
        return metrics

    def run_comprehensive_benchmark(
        self, df: pd.DataFrame, test_size: float = 0.20
    ) -> pd.DataFrame:
        """
        Executes comparison across:
        1. Majority Class Baseline (Zero-R)
        2. Rule-Based Heuristic Matcher
        3. VADER Sentiment Threshold Classifier
        4. Naive Bayes (Premise TF-IDF)
        5. Logistic Regression (Premise TF-IDF)
        6. Logistic Regression (Premise + Conclusion Context TF-IDF)
        7. Linear SVM (Premise + Conclusion Context TF-IDF)
        8. Random Forest (Engineered Dense Features)
        9. Hybrid Model (Linear SVM on TF-IDF + Engineered Dense Features)
        """
        # Ensure clean text and labels
        df = df[df["stance"].isin(["PRO", "CON"])].copy().reset_index(drop=True)
        
        # Pair text: Premise with Conclusion context
        df["text_with_context"] = (
            "Topic: " + df["conclusion"].fillna("") + " | Premise: " + df["premise_text"].fillna("")
        )

        # Stratified Train/Test Split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df["stance"],
        )

        y_train = train_df["stance"].values
        y_test = test_df["stance"].values

        # 1. Majority Baseline
        print("[1/9] Evaluating Majority Baseline (Zero-R)...")
        self.evaluate_model(
            "Majority Baseline (Zero-R)",
            DummyClassifier(strategy="most_frequent"),
            train_df["premise_text"],
            y_train,
            test_df["premise_text"],
            y_test,
        )

        # 2. Rule-Based Heuristic Matcher
        print("[2/9] Evaluating Rule-Based Heuristic Classifier...")
        self.evaluate_model(
            "Rule-Based Heuristic (Cues + Polarity)",
            RuleBasedStanceClassifier(use_vader=True),
            train_df["premise_text"],
            y_train,
            test_df["premise_text"],
            y_test,
            is_rule_based=True,
        )

        # 3. Pure VADER Threshold Classifier (Positive sentiment -> PRO, Negative -> CON)
        print("[3/9] Evaluating Pure VADER Sentiment Classifier...")
        class PureVaderClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.sia = SentimentIntensityAnalyzer()
                self.classes_ = np.array(["CON", "PRO"])
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                return np.array([
                    "PRO" if self.sia.polarity_scores(str(t))["compound"] >= 0.0 else "CON"
                    for t in X
                ])
        self.evaluate_model(
            "VADER Sentiment Polarity Baseline",
            PureVaderClassifier(),
            train_df["premise_text"],
            y_train,
            test_df["premise_text"],
            y_test,
            is_rule_based=True,
        )

        # Vectorizers for TF-IDF
        premise_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        X_train_premise_tfidf = premise_vectorizer.fit_transform(train_df["premise_text"].fillna(""))
        X_test_premise_tfidf = premise_vectorizer.transform(test_df["premise_text"].fillna(""))

        context_vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        X_train_ctx_tfidf = context_vectorizer.fit_transform(train_df["text_with_context"])
        X_test_ctx_tfidf = context_vectorizer.transform(test_df["text_with_context"])

        # 4. Multinomial Naive Bayes (Premise TF-IDF)
        print("[4/9] Evaluating Multinomial Naive Bayes (Premise TF-IDF)...")
        self.evaluate_model(
            "Multinomial Naive Bayes (Premise TF-IDF)",
            MultinomialNB(alpha=1.0),
            X_train_premise_tfidf,
            y_train,
            X_test_premise_tfidf,
            y_test,
        )

        # 5. Logistic Regression (Premise-only TF-IDF)
        print("[5/9] Evaluating Logistic Regression (Premise-only TF-IDF)...")
        self.evaluate_model(
            "Logistic Regression (Premise TF-IDF)",
            LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state),
            X_train_premise_tfidf,
            y_train,
            X_test_premise_tfidf,
            y_test,
        )

        # 6. Logistic Regression (Premise + Topic Context TF-IDF)
        print("[6/9] Evaluating Logistic Regression (Contextual TF-IDF)...")
        self.evaluate_model(
            "Logistic Regression (Contextual TF-IDF)",
            LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state),
            X_train_ctx_tfidf,
            y_train,
            X_test_ctx_tfidf,
            y_test,
        )

        # 7. Linear SVM (Contextual TF-IDF)
        print("[7/9] Evaluating Linear SVM (Contextual TF-IDF)...")
        self.evaluate_model(
            "Linear SVM (Contextual TF-IDF)",
            LinearSVC(C=0.5, max_iter=2000, random_state=self.random_state),
            X_train_ctx_tfidf,
            y_train,
            X_test_ctx_tfidf,
            y_test,
        )

        # Dense Engineered Features
        print("[8/9] Extracting Dense Linguistic Features & Evaluating Random Forest...")
        from sklearn.preprocessing import StandardScaler
        X_train_dense_raw = self.dense_extractor.extract_features_df(train_df)
        X_test_dense_raw = self.dense_extractor.extract_features_df(test_df)

        self.evaluate_model(
            "Random Forest (Linguistic & Stylistic Features)",
            RandomForestClassifier(n_estimators=100, max_depth=12, random_state=self.random_state),
            X_train_dense_raw,
            y_train,
            X_test_dense_raw,
            y_test,
        )

        # 9. Hybrid Model: Scale Dense Features and Stack with Contextual TF-IDF
        print("[9/9] Evaluating Hybrid Classifier (TF-IDF + Linguistic Dense Features)...")
        scaler = StandardScaler()
        X_train_dense_scaled = scaler.fit_transform(X_train_dense_raw)
        X_test_dense_scaled = scaler.transform(X_test_dense_raw)

        X_train_hybrid = hstack([X_train_ctx_tfidf, csr_matrix(X_train_dense_scaled)])
        X_test_hybrid = hstack([X_test_ctx_tfidf, csr_matrix(X_test_dense_scaled)])

        self.evaluate_model(
            "Hybrid Model (Linear SVM + TF-IDF + Dense Features)",
            LinearSVC(C=0.5, max_iter=3000, dual="auto", random_state=self.random_state),
            X_train_hybrid,
            y_train,
            X_test_hybrid,
            y_test,
        )

        # Compile comparison DataFrame
        summary_rows = []
        for r in self.results:
            summary_rows.append({
                "Model": r.model_name,
                "Accuracy": r.accuracy,
                "Macro F1": r.macro_f1,
                "Macro Precision": r.macro_precision,
                "Macro Recall": r.macro_recall,
                "CON F1": r.con_f1,
                "PRO F1": r.pro_f1,
                "ROC-AUC": r.roc_auc if r.roc_auc is not None else "N/A",
            })

        return pd.DataFrame(summary_rows)

    def extract_top_features(
        self,
        df: pd.DataFrame,
        n_top: int = 15,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Fit Logistic Regression on contextual TF-IDF and return
        the top indicative n-grams for PRO and CON.
        """
        df_clean = df[df["stance"].isin(["PRO", "CON"])].copy()
        text_ctx = "Topic: " + df_clean["conclusion"].fillna("") + " | Premise: " + df_clean["premise_text"].fillna("")
        y = (df_clean["stance"] == "PRO").astype(int).values

        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        X = vectorizer.fit_transform(text_ctx)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        clf.fit(X, y)

        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = clf.coef_[0]

        top_con_indices = np.argsort(coefs)[:n_top]
        top_pro_indices = np.argsort(coefs)[-n_top:][::-1]

        top_con = [(feature_names[i], float(coefs[i])) for i in top_con_indices]
        top_pro = [(feature_names[i], float(coefs[i])) for i in top_pro_indices]

        return top_pro, top_con

    def run_cross_topic_generalization(
        self, df: pd.DataFrame
    ) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        """
        Evaluate model generalization when tested on UNSEEN topics
        (GroupShuffleSplit by 'conclusion') vs random in-topic split.
        Reveals whether traditional models learn true stance or topic vocabulary.
        """
        df_clean = df[df["stance"].isin(["PRO", "CON"])].copy().reset_index(drop=True)
        text_ctx = "Topic: " + df_clean["conclusion"].fillna("") + " | Premise: " + df_clean["premise_text"].fillna("")
        y = df_clean["stance"].values
        groups = df_clean["conclusion"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(text_ctx, y, groups=groups))

        vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        X_train = vec.fit_transform(text_ctx.iloc[train_idx])
        X_test = vec.transform(text_ctx.iloc[test_idx])

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)

        acc = float(accuracy_score(y[test_idx], preds))
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], preds, average="macro", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y[test_idx], preds, labels=["CON", "PRO"], zero_division=0)

        out_of_topic_metrics = EvaluationMetrics(
            model_name="Out-of-Topic Logistic Regression (Unseen Topics)",
            accuracy=round(acc, 4),
            macro_f1=round(f1, 4),
            macro_precision=round(p, 4),
            macro_recall=round(r, 4),
            con_f1=round(f1_c[0], 4),
            pro_f1=round(f1_c[1], 4),
        )
        return out_of_topic_metrics
