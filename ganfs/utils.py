"""
Utility functions for GANFS.

Provides GPU configuration and data preprocessing helpers used
throughout the GANFS pipeline.
"""

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("ganfs")


def setup_gpu():
    """Detect and configure GPU with memory growth enabled.

    Enables memory growth on all available GPUs so TensorFlow does not
    pre-allocate the entire GPU memory. Returns the appropriate device
    string for ``tf.device()``.

    Returns
    -------
    str
        Device string: ``'/GPU:0'`` if a GPU is available, ``'/CPU:0'`` otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU detected and memory growth enabled.")
        except RuntimeError as e:
            logger.warning("Error setting GPU memory growth: %s", e)
        return '/GPU:0'
    else:
        logger.info("No GPU detected; running on CPU.")
        return '/CPU:0'


def preprocess_dataframe(df, label_col='Label'):
    """Preprocess a DataFrame for GANFS.

    Applies the same preprocessing pipeline used in the original GANFS
    research:

    1. Replace infinite values with 0
    2. Coerce non-numeric values to NaN, then fill with 0
    3. Separate features (X) and labels (y)
    4. MinMax-scale features to [0, 1]

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and a label column.
    label_col : str, default ``'Label'``
        Name of the label/target column.

    Returns
    -------
    X_scaled : np.ndarray
        MinMax-scaled feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Label array of shape ``(n_samples,)``.
    feature_names : list[str]
        List of feature column names (in order).
    scaler : MinMaxScaler
        Fitted scaler instance (for transforming new data).

    Raises
    ------
    KeyError
        If ``label_col`` is not found in the DataFrame.
    """
    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    if label_col not in df.columns:
        raise KeyError(
            f"Column '{label_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Replace infinities
    df = df.replace([np.inf, -np.inf, 'Infinity'], 0)

    # Separate features and labels
    y = df[label_col].values
    X_df = df.drop(columns=[label_col])

    # Keep only numeric columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    X_df = X_df[numeric_cols]

    # Coerce remaining non-numeric values
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    # MinMax scale to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names, scaler
