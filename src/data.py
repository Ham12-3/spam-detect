"""
Data loading and validation module for SMS Spam Detector.

Handles loading the SMS Spam Collection dataset and validating its structure.
"""

import os
import hashlib
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


# Expected path for the dataset
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "sms_spam.csv"

# Required columns in the dataset
REQUIRED_COLUMNS = {"label", "text"}
VALID_LABELS = {"spam", "ham"}

# Common column name mappings (source -> target)
# Handles various dataset formats automatically
COLUMN_MAPPINGS = {
    "v1": "label",
    "v2": "text",
    "class": "label",
    "message": "text",
    "sms": "text",
    "category": "label",
    "Label": "label",
    "Text": "text",
    "MESSAGE": "text",
    "LABEL": "label",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names to expected format.

    Handles common variations like v1/v2 from Kaggle dataset.

    Args:
        df: DataFrame with potentially non-standard column names.

    Returns:
        DataFrame with normalised column names.
    """
    df = df.copy()

    # Apply known mappings
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPINGS:
            rename_map[col] = COLUMN_MAPPINGS[col]

    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop any unnamed or empty columns (common in Kaggle exports)
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def get_file_hash(filepath: Path) -> str:
    """
    Compute MD5 hash of a file for cache invalidation.

    Args:
        filepath: Path to the file.

    Returns:
        MD5 hash string of the file contents.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that a DataFrame has the required structure for the spam detector.

    Args:
        df: DataFrame to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Check for required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}. Expected: {REQUIRED_COLUMNS}"

    # Check for empty DataFrame
    if len(df) == 0:
        return False, "Dataset is empty."

    # Normalise labels to lowercase and check validity
    df["label"] = df["label"].str.lower().str.strip()
    unique_labels = set(df["label"].unique())
    invalid_labels = unique_labels - VALID_LABELS

    if invalid_labels:
        return False, f"Invalid labels found: {invalid_labels}. Expected only: {VALID_LABELS}"

    # Check for missing values
    null_counts = df[["label", "text"]].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        return False, f"Null values found in columns: {cols_with_nulls}"

    # Check for empty text
    empty_text_count = (df["text"].str.strip() == "").sum()
    if empty_text_count > 0:
        return False, f"Found {empty_text_count} rows with empty text."

    return True, None


def load_dataset(filepath: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Load and validate the SMS spam dataset.

    Args:
        filepath: Optional custom path to dataset. Uses default if None.

    Returns:
        Tuple of (dataframe, error_message, file_hash).
        If loading fails, dataframe is None and error_message explains why.
    """
    path = filepath or DEFAULT_DATASET_PATH

    # Check if file exists
    if not path.exists():
        error_msg = f"""
Dataset file not found at: {path}

Please download the SMS Spam Collection dataset and save it as a CSV file.

**Expected location:** `{path}`

**Expected format:**
- CSV file with columns: `label`, `text`
- `label` column should contain: "spam" or "ham"
- `text` column should contain the SMS message

**Example rows:**
```
label,text
ham,Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam,Free entry in 2 a wkly comp to win FA Cup final tkts...
```

You can download the dataset from:
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
        """
        return None, error_msg.strip(), None

    try:
        # Try different encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return None, f"Could not decode file with common encodings.", None

        # Normalise column names (handles v1/v2 from Kaggle, etc.)
        df = normalise_columns(df)

        # Validate structure
        is_valid, error = validate_dataframe(df)
        if not is_valid:
            return None, error, None

        # Normalise the data
        df["label"] = df["label"].str.lower().str.strip()
        df["text"] = df["text"].astype(str)

        # Compute file hash for caching
        file_hash = get_file_hash(path)

        return df, None, file_hash

    except pd.errors.EmptyDataError:
        return None, "Dataset file is empty.", None
    except pd.errors.ParserError as e:
        return None, f"Error parsing CSV file: {e}", None
    except Exception as e:
        return None, f"Unexpected error loading dataset: {e}", None


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute statistics about the dataset.

    Args:
        df: The loaded DataFrame.

    Returns:
        Dictionary with dataset statistics.
    """
    label_counts = df["label"].value_counts()

    return {
        "total_samples": len(df),
        "ham_count": int(label_counts.get("ham", 0)),
        "spam_count": int(label_counts.get("spam", 0)),
        "ham_percentage": round(label_counts.get("ham", 0) / len(df) * 100, 1),
        "spam_percentage": round(label_counts.get("spam", 0) / len(df) * 100, 1),
        "avg_message_length": round(df["text"].str.len().mean(), 1),
        "min_message_length": int(df["text"].str.len().min()),
        "max_message_length": int(df["text"].str.len().max()),
    }
