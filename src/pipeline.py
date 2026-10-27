# importing the necessary libraries
from __future__ import annotations
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# resolving the path to the artifacts folder relative to this script
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# loading the saved pipeline (preprocessing + tuned CatBoost in one object) and its metadata
lead_pipeline = joblib.load(filename = ARTIFACT_DIR / "lead_evaluator_pipeline.joblib")
metadata = joblib.load(filename = ARTIFACT_DIR / "metadata.joblib")

# extracting metadata into module-level constants
INPUT_COLUMNS = metadata["input_columns"]
NUMERIC_COLS = metadata["numeric_cols"]
CATEGORY_LEVELS = metadata["category_levels"]
NUMERIC_RANGES = metadata["numeric_ranges"]
THR_PRIORITIZE = metadata["qualified_thresholds"]["prioritize"]
THR_NURTURE = metadata["qualified_thresholds"]["nurture"]

# spellings of Yes/No that new files may use (True/False, 1/0, yes/no in any case); the model was trained on "Yes"/"No" text
YES_NO_LEVELS = {"Yes", "No"}
YES_NO_MAP = {"yes": "Yes", "true": "Yes", "1": "Yes", "1.0": "Yes", "no": "No", "false": "No", "0": "No", "0.0": "No"}


# creating the helper functions

# selecting the model inputs from the raw leads and fixing their types (the preprocessing itself lives inside the saved pipeline)
def _prepare_inputs(raw_df: pd.DataFrame) -> pd.DataFrame:

    # keeping only the columns the model was trained on (company_name, lead_bucket and other extras are ignored)
    X = raw_df[INPUT_COLUMNS].copy()

    # turning text or invalid numbers into missing values, so that they are flagged instead of crashing
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors = "coerce")

    # turning every categorical column into clean text, because the encoders were fitted on text (numbers or True/False would crash them or be mistaken for new categories)
    for col, levels in CATEGORY_LEVELS.items():
        text = X[col].astype(object)
        present = text.notna()
        text[present] = text[present].astype(str).str.strip()

        # Yes/No columns (e.g. has_phone): True/False, 1/0 and yes/no are all converted back to "Yes"/"No"
        if set(levels) <= YES_NO_LEVELS:
            text[present] = text[present].str.lower().map(YES_NO_MAP).fillna(text[present])

        X[col] = text

    return X


# checking every lead before scoring: missing or impossible values block the scoring, odd-but-usable values only add a note
def check_leads(raw_df: pd.DataFrame) -> pd.DataFrame:

    X = _prepare_inputs(raw_df)
    blocked = pd.Series(False, index = X.index)
    notes = [[] for _ in range(len(X))]

    # 1. missing values: CatBoost would silently treat a missing number as the smallest value (for urgency_days that means "most urgent"), so these leads are not scored
    for col in X.columns:
        missing = X[col].isna().to_numpy()
        blocked |= missing
        for i in np.flatnonzero(missing):
            notes[i].append(f"{col} missing")

    # 2. impossible values: none of the numeric fields can be negative
    for col in NUMERIC_COLS:
        negative = (X[col] < 0).fillna(False).to_numpy()
        blocked |= negative
        for i in np.flatnonzero(negative):
            notes[i].append(f"{col} negative")

    # 3. values outside the range seen in training: the lead is still scored, but the model is extrapolating
    for col in NUMERIC_COLS:
        low, high = NUMERIC_RANGES[col]
        outside = (((X[col] < low) & (X[col] >= 0)) | (X[col] > high)).fillna(False).to_numpy()
        for i in np.flatnonzero(outside):
            notes[i].append(f"{col} outside training range ({low:g} to {high:g})")

    # 4. categories never seen in training: the encoders silently ignore them
    for col, levels in CATEGORY_LEVELS.items():
        unseen = (~X[col].isin(levels) & X[col].notna()).to_numpy()
        for i in np.flatnonzero(unseen):
            notes[i].append(f"unseen {col} value '{X[col].iloc[i]}'")

    # 5. zero volume usually means "volume not provided" (most such leads still order weekly or monthly)
    volume_col = "expected_monthly_volume_liters"
    if volume_col in X.columns:
        for i in np.flatnonzero((X[volume_col] == 0).fillna(False).to_numpy()):
            notes[i].append("volume not provided (0 liters)")

    return pd.DataFrame(
        {"blocked": blocked.to_numpy(), "data_flag": ["; ".join(n) for n in notes]},
        index = raw_df.index
    )

# mapping a qualifiedScore (0–100) to a business action: >= 70: Prioritize (Sales); >= 40: Nurture (Follow-up); <  40: Deprioritize
def score_to_action(score: float) -> str:
    if score >= THR_PRIORITIZE:
        return "Prioritize (Sales)"
    if score >= THR_NURTURE:
        return "Nurture (Follow-up)"
    return "Deprioritize"


# creating the main prediction function

# scoring new leads end-to-end: check inputs → predict probabilities → pick bucket → compute QualifiedScore → assign action
def predict_leads(raw_df: pd.DataFrame) -> pd.DataFrame:

    # stopping early if a required column is absent from the file
    missing_cols = [c for c in INPUT_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required column(s): {missing_cols}")

    # flagging leads that cannot be scored safely
    checks = check_leads(raw_df)
    scorable = ~checks["blocked"].to_numpy()

    # starting with "not scored" for every lead, then filling in the scorable ones
    pred_bucket = np.full(len(raw_df), "Needs review", dtype = object)
    lead_score = np.full(len(raw_df), np.nan)
    action = np.full(len(raw_df), "Check data first", dtype = object)

    if scorable.any():

        # predicting class probabilities with the saved pipeline (it encodes the raw columns itself)
        X = _prepare_inputs(raw_df).loc[scorable]
        with warnings.catch_warnings():
            warnings.filterwarnings(action = "ignore", message = "Found unknown categories")
            proba = lead_pipeline.predict_proba(X)

        # retrieving class labels and the Hot and Warm column indices
        classes = np.array(lead_pipeline.classes_)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        hot_idx = class_to_idx["Hot"]
        warm_idx = class_to_idx["Warm"]

        # predicting the most likely lead bucket
        pred_bucket[scorable] = classes[np.argmax(proba, axis = 1)]

        # computing QualifiedScore = 100 × (P(Hot) + P(Warm))
        scores = np.round(100.0 * (proba[:, hot_idx] + proba[:, warm_idx]), 1)
        lead_score[scorable] = scores

        # mapping each score to a business action
        action[scorable] = [score_to_action(s) for s in scores]

    # appending predictions to a copy of the original dataframe
    out = raw_df.copy()
    out["pred_bucket"] = pred_bucket
    out["lead_score"] = lead_score
    out["action"] = action
    out["data_flag"] = checks["data_flag"].to_numpy()

    return out
