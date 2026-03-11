# importing the necessary libraries
from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# resolving the path to the artifacts folder relative to this script
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# loading the saved artifacts from the artifacts directory
cat_model = joblib.load(filename = ARTIFACT_DIR / "catboost_tuned.joblib")
ordinal_encoder = joblib.load(filename = ARTIFACT_DIR / "ordinal_encoder.joblib")
onehot_encoder = joblib.load(filename = ARTIFACT_DIR / "onehot_encoder.joblib")
scaler = joblib.load(filename = ARTIFACT_DIR / "scaler.joblib")
metadata = joblib.load(filename = ARTIFACT_DIR / "metadata.joblib")

# extracting metadata into module-level constants
ORDINAL_COLS = metadata["ordinal_cols"]
ONEHOT_COLS = metadata["onehot_cols"]
NUMERIC_COLS = metadata["numeric_cols"]
FEATURE_COLUMNS = metadata["feature_columns"]
EMP_ORDER = metadata["employee_order"]
CRED_ORDER = metadata["credibility_order"]
PURCHASE_ORDER = metadata["purchase_stage_order"]
YN_MAP = metadata["yn_map"]
THR_PRIORITIZE = metadata["qualified_thresholds"]["prioritize"]
THR_NURTURE = metadata["qualified_thresholds"]["nurture"]


# creating the helper functions

# converting a Yes/No or True/False series to integer (1/0)
def _yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.map(YN_MAP).fillna(0).astype(int)

# applying the exact same preprocessing pipeline used during training: drop non-features → boolean conversion → ordinal encoding → one-hot encoding → scaling → column alignment
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:

    # creating a copy to avoid mutating the original dataframe
    df = raw_df.copy()

    # 1. dropping non-feature columns
    drop_cols = ["company_name", "lead_bucket", "lead_score", "actual_lead_score"]
    df = df.drop(columns = [c for c in drop_cols if c in df.columns])

    # 2. converting Yes/No boolean fields to 1/0
    for col in ["has_google_listing", "has_phone"]:
        if col in df.columns:
            df[col] = _yes_no_to_int(df[col])

    # 3. setting ordinal categories (same order as training)
    df["employee_estimate"] = pd.Categorical(df["employee_estimate"], categories = EMP_ORDER, ordered = True)
    df["credibility_level"] = pd.Categorical(df["credibility_level"], categories = CRED_ORDER, ordered = True)
    df["purchase_stage"] = pd.Categorical(df["purchase_stage"], categories = PURCHASE_ORDER, ordered = True)

    # 4. ordinal encoding
    df[ORDINAL_COLS] = ordinal_encoder.transform(df[ORDINAL_COLS])

    # 5. one-hot encoding
    oh = pd.DataFrame(
        onehot_encoder.transform(df[ONEHOT_COLS]),
        columns = onehot_encoder.get_feature_names_out(ONEHOT_COLS),
        index = df.index,
    )
    df = df.drop(columns = ONEHOT_COLS).join(oh)

    # 6. standard scaling on numeric columns
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # 7. reindex to match exact training column order (any missing columns filled with 0, any extra columns dropped)
    df = df.reindex(columns = FEATURE_COLUMNS, fill_value = 0)

    return df

# mapping a qualifiedScore (0–100) to a business action: >= 70: Prioritize (Sales); >= 40: Nurture (Follow-up); <  40: Deprioritize
def score_to_action(score: float) -> str:
    if score >= THR_PRIORITIZE:
        return "Prioritize (Sales)"
    if score >= THR_NURTURE:
        return "Nurture (Follow-up)"
    return "Deprioritize"



# creating the main prediction function

# scoring new leads end-to-end: preprocess → predict bucket → compute QualifiedScore → assign action.
def predict_leads(raw_df: pd.DataFrame) -> pd.DataFrame:

    # preprocessing the raw input dataframe
    X = preprocess(raw_df)

    # generating predicted class probabilities
    proba = cat_model.predict_proba(X)

    # retrieving class labels
    classes = cat_model.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # finding Hot and Warm column indices
    hot_idx = class_to_idx["Hot"]
    warm_idx = class_to_idx["Warm"]

    # predicting the most likely lead bucket
    pred_bucket = np.array(cat_model.predict(X)).ravel()

    # computing QualifiedScore = 100 × (P(Hot) + P(Warm))
    lead_score = 100.0 * (proba[:, hot_idx] + proba[:, warm_idx])
    lead_score = np.round(lead_score, 1)

    # mapping each score to a business action
    action = np.array([score_to_action(s) for s in lead_score])

    # appending predictions to a copy of the original dataframe
    out = raw_df.copy()
    out["pred_bucket"] = pred_bucket
    out["lead_score"] = lead_score
    out["action"] = action

    return out