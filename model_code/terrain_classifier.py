import os
from lightgbm import LGBMClassifier
import joblib
import lunadem
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler
from sklearn.svm import SVC


RAW_FEATURES = [
    "solar_zenith",
    "surface_temp",
    "elevation",
    "slope",
    "reflectance",
    "crater_density",
    "sensor_noise_alpha",
    "sensor_noise_beta",
]

DERIVED_FEATURES = [
    "mineral_index",
    "thermal_inertia",
    "albedo_ratio",
    "regolith_depth",
]

INSIGHT_FEATURES = [
    "slope_gt_13",
    "slope_gt_14",
    "slope_log",
    "slope_sqrt",
    "slope_sq",
    "slope_margin_13",
    "reflectance_log",
    "elevation_abs",
    "slope_x_reflectance",
    "slope_x_crater_density",
]

FEATURES = RAW_FEATURES + DERIVED_FEATURES + INSIGHT_FEATURES
FEATURE_GROUPS = {
    "raw": RAW_FEATURES,
    "derived": DERIVED_FEATURES,
    "insight": INSIGHT_FEATURES,
}
WEIGHTS_PATH = os.path.join("weights_file", "model.pkl")


def add_derived_features(df):
    """Return a copy of df with LinaDEM and distribution-aware features present."""
    enriched = df.copy()
    enriched["mineral_index"] = lunadem.extract_feature_alpha(enriched)
    enriched["thermal_inertia"] = lunadem.extract_feature_beta(enriched)
    enriched["albedo_ratio"] = lunadem.extract_feature_gamma(enriched)
    enriched["regolith_depth"] = lunadem.extract_feature_delta(enriched)
    enriched["slope_gt_13"] = (enriched["slope"] > 13).astype(int)
    enriched["slope_gt_14"] = (enriched["slope"] > 14).astype(int)
    enriched["slope_log"] = np.log1p(enriched["slope"])
    enriched["slope_sqrt"] = np.sqrt(enriched["slope"])
    enriched["slope_sq"] = enriched["slope"] ** 2
    enriched["slope_margin_13"] = 13 - enriched["slope"]
    enriched["reflectance_log"] = np.log(enriched["reflectance"])
    enriched["elevation_abs"] = enriched["elevation"].abs()
    enriched["slope_x_reflectance"] = enriched["slope"] * enriched["reflectance"]
    enriched["slope_x_crater_density"] = enriched["slope"] * enriched["crater_density"]
    return enriched


def apply_feature_weights(X, feature_weights=None):
    if feature_weights is None:
        return X

    weights = pd.Series(1.0, index=X.columns)

    # Group weights are applied first, then individual feature weights override them.
    for key, value in feature_weights.items():
        if key in FEATURE_GROUPS:
            for feature in FEATURE_GROUPS[key]:
                if feature in weights.index:
                    weights.loc[feature] = float(value)

    for key, value in feature_weights.items():
        if key in weights.index:
            weights.loc[key] = float(value)

    return X.mul(weights, axis=1)


def prepare_features(df, feature_weights=None):
    enriched = add_derived_features(df)
    missing = [feature for feature in FEATURES if feature not in enriched.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    X = enriched[FEATURES]
    return apply_feature_weights(X, feature_weights)


def build_model():
    quantile_logistic = Pipeline(
        [
            (
                "quantile",
                QuantileTransformer(output_distribution="uniform", random_state=42),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=5000, C=50.0, random_state=42),
            ),
        ]
    )
    spline_logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "spline",
                SplineTransformer(n_knots=7, degree=5, include_bias=True),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=5000, C=0.1, random_state=42),
            ),
        ]
    )
    neural_net = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(256),  # Increased capacity
                    activation="logistic",  # More complex activation for non-linearity
                    solver="adam",
                    alpha=0.001,  # Slightly lower regularization to allow more learning
                    learning_rate="adaptive",  # Keeps learning efficient
                    learning_rate_init=0.001,
                    max_iter=3000,  # Given more time to converge
                    random_state=42,
                    early_stopping=False,
                ),
            ),
        ]
    )
    support_vector = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(C=10.0, gamma="scale", probability=True, random_state=42),
            ),
        ]
    )
    lightgbm_model = LGBMClassifier(
        n_estimators=150,
        num_leaves=31,
        max_bin=255,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,  # Keeps the console clean from training logs
    )
    # boosted_trees = AdaBoostClassifier(n_estimators=200, random_state=42)

    return VotingClassifier(
        estimators=[
            ("quantile_logistic", quantile_logistic),
            ("spline_logistic", spline_logistic),
            ("neural_net", neural_net),
            ("support_vector", support_vector),
            # ("boosted_trees", boosted_trees),
            ("lightgbm", lightgbm_model),
        ],
        voting="soft",
        n_jobs=-1,
    )


def train_model(weights_path=WEIGHTS_PATH, feature_weights=None):
    print("Loading historical LinaDEM data...")
    df = lunadem.get_previously_available_data()
    hypothesis_weights = {
        "insight": 0.001,
        "solar_zenith": 0.001,
        "surface_temp": 0.001,
        "elevation": 0.001,
        "slope": 10.0,
        "reflectance": 0.001,
        "crater_density": 0.001,
        "sensor_noise_alpha": 0.001,
        "sensor_noise_beta": 0.001,
        "mineral_index": 0.001,
        "thermal_inertia": 10.0,
        "albedo_ratio": 0.01,
        "regolith_depth": 10.0,
    }
    weights = feature_weights if feature_weights is not None else hypothesis_weights
    X = prepare_features(df, feature_weights=weights)
    y = df["label"].astype(int)

    X_train, X_valid, y_train, y_valid, raw_train, raw_valid = train_test_split(
        X,
        y,
        df,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training probability ensemble terrain classifier...")
    model = build_model()
    model.fit(X_train, y_train)

    valid_probabilities = predict_dataframe(raw_valid, model=model, feature_weights=weights)
    valid_predictions = (valid_probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(y_valid, valid_predictions)
    roc_auc = roc_auc_score(y_valid, valid_probabilities)
    reference_labels = _predict_reference_labels(raw_valid)
    reference_accuracy = accuracy_score(reference_labels, valid_predictions)
    reference_roc_auc = roc_auc_score(reference_labels, valid_probabilities)
    matrix = confusion_matrix(y_valid, valid_predictions, labels=[0, 1])

    print(f"Holdout accuracy: {accuracy:.4f}")
    print(f"Holdout ROC-AUC: {roc_auc:.4f}")
    print(f"Reference engine accuracy: {reference_accuracy:.4f}")
    print(f"Reference engine ROC-AUC: {reference_roc_auc:.4f}")
    print("Confusion matrix [[tn, fp], [fn, tp]]:")
    print(matrix)

    print("Retraining on all historical data...")
    final_model = build_model()
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    joblib.dump(final_model, weights_path)
    print(f"Model saved to {weights_path}")
    return final_model


def load_model(weights_path=WEIGHTS_PATH):
    return joblib.load(weights_path)


def predict_dataframe(df, model=None, feature_weights=None):
    if model is None:
        model = load_model()
    X = prepare_features(df, feature_weights=feature_weights)
    class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def _predict_reference_labels(df):
    enriched = add_derived_features(df)
    required_features = RAW_FEATURES + DERIVED_FEATURES
    return pd.Series(
        [lunadem.predict_label(enriched.iloc[index][required_features]) for index in range(len(enriched))],
        index=enriched.index,
    )
