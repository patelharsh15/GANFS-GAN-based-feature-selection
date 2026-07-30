"""
GANFS — GAN-based Feature Selection.

Main module providing the :class:`GANFS` class, which wraps the entire
GAN training and sensitivity-analysis pipeline into a scikit-learn-style
estimator with ``fit``, ``transform``, and ``fit_transform`` methods.
"""

import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
# Use tf.keras directly to avoid conflicts with standalone keras / JAX installations
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers

from ganfs.models import build_discriminator, build_generator
from ganfs.sensitivity import (
    compute_baseline_predictions,
    feature_pair_sensitivity,
    sensitivity_analysis,
)
from ganfs.utils import preprocess_dataframe, setup_gpu

logger = logging.getLogger("ganfs")


class GANFS(BaseEstimator, TransformerMixin):
    """GAN-based Feature Selection.

    Trains a Generative Adversarial Network on the input data and uses
    perturbation-based sensitivity analysis on the discriminator to rank
    features by importance.

    Parameters
    ----------
    epochs : int, default 500
        Number of GAN training epochs.
    batch_size : int, default 4096
        Batch size for GAN training.
    learning_rate : float, default 0.001
        Learning rate for Adam optimizers (both generator and discriminator).
    label_smoothing : tuple[float, float], default (0.9, 0.1)
        Label smoothing values for (real, fake) labels during
        discriminator training. Improves training stability.
    perturbation_mode : str, default ``'dynamic'``
        ``'dynamic'`` scales perturbations by each feature's natural
        granularity; ``'static'`` uses raw perturbation factors.
    perturbation_factors : list[float] or None
        Perturbation multipliers/deltas for sensitivity analysis.
        Defaults to ``[0.5, 1.0, 2.0, 5.0, 10.0]``.
    inference_batch_size : int or None
        Batch size for sensitivity analysis inference. Defaults to
        ``batch_size // 2`` if None.
    checkpoint_dir : str or None
        Directory to save training checkpoints. Set to None to disable.
    verbose : bool, default True
        If True, log progress information during training and analysis.
    random_state : int or None, default None
        Random seed for reproducibility.
    patience : int or None, default None
        Number of epochs to wait for discriminator loss improvement before
        stopping training early. If None, trains for all `epochs`.
    generator_hidden_layers : tuple[int], default (64, 128)
        Sizes of the hidden Dense layers for the generator.
    discriminator_hidden_layers : tuple[int], default (128, 64)
        Sizes of the hidden Dense layers for the discriminator.

    Attributes
    ----------
    generator_ : tf.keras.Model
        Trained generator model (available after ``fit``).
    discriminator_ : tf.keras.Model
        Trained discriminator model (available after ``fit``).
    feature_names_ : list[str]
        Feature names from the training data.
    sensitivities_ : np.ndarray
        Per-feature sensitivity scores (available after ``fit``).
    sorted_indices_ : np.ndarray
        Feature indices sorted by sensitivity (descending).
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> from ganfs import GANFS
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("data.csv")
    >>> X = df.drop("label", axis=1)
    >>> y = df["label"]
    >>>
    >>> selector = GANFS(epochs=100, verbose=True)
    >>> selector.fit(X, y)
    >>> ranking = selector.get_feature_ranking()
    >>> X_top20 = selector.transform(X, k=20)
    """

    def __init__(
        self,
        epochs=500,
        batch_size=4096,
        learning_rate=0.001,
        label_smoothing=(0.9, 0.1),
        perturbation_mode='dynamic',
        perturbation_factors=None,
        inference_batch_size=None,
        checkpoint_dir=None,
        verbose=True,
        random_state=None,
        patience=None,
        generator_hidden_layers=(64, 128),
        discriminator_hidden_layers=(128, 64),
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.perturbation_mode = perturbation_mode
        self.perturbation_factors = perturbation_factors or [0.5, 1.0, 2.0, 5.0, 10.0]
        self.inference_batch_size = inference_batch_size
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.random_state = random_state
        self.patience = patience
        self.generator_hidden_layers = generator_hidden_layers
        self.discriminator_hidden_layers = discriminator_hidden_layers

        # Internal state (set during fit)
        self.generator_ = None
        self.discriminator_ = None
        self.gan_ = None
        self.feature_names_ = None
        self.scaler_ = None
        self.sensitivities_ = None
        self.sorted_indices_ = None
        self.pair_results_ = None
        self.is_fitted_ = False
        self._device = None
        self._n_features = None

    def fit(self, X, y=None, label_col='Label'):
        """Train the GAN and compute feature sensitivity scores.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix. If a DataFrame, column names are preserved.
            If ``y`` is None and ``X`` is a DataFrame, it should contain
            the label column specified by ``label_col``.
        y : array-like or None, default None
            Target labels. If None and ``X`` is a DataFrame, the label
            column is extracted from ``X`` using ``label_col``.
        label_col : str, default ``'Label'``
            Name of the label column if ``X`` is a DataFrame and ``y``
            is None. Ignored if ``y`` is provided.

        Returns
        -------
        self
            Fitted GANFS instance.
        """
        if self.random_state is not None:
            import random
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
            # Use keras native seed setter if available
            try:
                tf.keras.utils.set_random_seed(self.random_state)
            except AttributeError:
                pass

        # Configure logging
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [GANFS] %(message)s',
                datefmt='%H:%M:%S'
            )

        # Setup GPU
        self._device = setup_gpu()

        # Preprocess data
        X_scaled, feature_names = self._prepare_data(X, y, label_col)
        self.feature_names_ = feature_names
        self._n_features = X_scaled.shape[1]

        if self.verbose:
            logger.info(
                "Data prepared: %d samples, %d features",
                X_scaled.shape[0], self._n_features
            )

        # Build models
        self._build_models()

        # Train GAN
        self._train_gan(X_scaled)

        # Compute sensitivity
        inf_batch = self.inference_batch_size or max(self.batch_size // 2, 1024)
        self._compute_sensitivity(X_scaled, inf_batch)

        self.is_fitted_ = True

        if self.verbose:
            logger.info("GANFS fitting complete!")
            top5 = self.get_feature_ranking().head(5)
            logger.info("Top 5 features:\n%s", top5.to_string(index=False))

        return self

    def transform(self, X, k=None):
        """Select the top-K features from X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix to transform.
        k : int or None
            Number of top features to select. If None, returns all
            features sorted by importance.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Transformed data with only the selected features.
            Returns a DataFrame if input is a DataFrame, otherwise ndarray.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        if k is None:
            k = self._n_features

        top_indices = self.sorted_indices_[:k]
        top_names = [self.feature_names_[i] for i in top_indices]

        if isinstance(X, pd.DataFrame):
            # Strip column names to match
            X = X.copy()
            X.columns = X.columns.str.strip()
            available = [n for n in top_names if n in X.columns]
            return X[available]
        else:
            return X[:, top_indices]

    def fit_transform(self, X, y=None, k=None, label_col='Label'):
        """Fit the model and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (with or without label column).
        y : array-like or None
            Target labels.
        k : int or None
            Number of top features to select.
        label_col : str, default ``'Label'``
            Label column name (if ``y`` is None).

        Returns
        -------
        pd.DataFrame or np.ndarray
            Transformed data with only the selected features.
        """
        self.fit(X, y, label_col=label_col)
        return self.transform(X, k=k)

    def get_feature_ranking(self):
        """Get features ranked by sensitivity score.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``'Rank'``, ``'Feature'``, and
            ``'Sensitivity_Score'``, sorted by sensitivity (descending).

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        ranking = pd.DataFrame({
            'Rank': range(1, len(self.feature_names_) + 1),
            'Feature': [self.feature_names_[i] for i in self.sorted_indices_],
            'Sensitivity_Score': self.sensitivities_[self.sorted_indices_]
        })
        return ranking

    def get_feature_pairs(self, top_n=20, delta=0.1):
        """Analyze synergistic interactions between top feature pairs.

        Computes pairwise perturbation effects for the top-N most
        sensitive features and identifies pairs with super-additive
        (synergistic) effects.

        Parameters
        ----------
        top_n : int, default 20
            Number of top features to consider.
        delta : float, default 0.1
            Perturbation magnitude for pair analysis.

        Returns
        -------
        pd.DataFrame
            DataFrame with pair interaction results, sorted by synergy score.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        if self.pair_results_ is not None:
            return self._pair_results_to_df(self.pair_results_)

        inf_batch = self.inference_batch_size or max(self.batch_size // 2, 1024)

        if self.verbose:
            logger.info("Computing feature pair interactions (top_n=%d)...", top_n)

        # We need X_scaled — recompute baseline from stored model
        # The user should call this during/after fit when data is available
        warnings.warn(
            "Pair analysis requires the original training data. "
            "Call compute_pairs(X_scaled, baseline_preds) internally during fit, "
            "or provide data via get_feature_pairs_from_data().",
            stacklevel=2
        )
        return pd.DataFrame()

    def get_feature_pairs_from_data(self, X, top_n=20, delta=0.1):
        """Analyze feature pair interactions using provided data.

        Parameters
        ----------
        X : np.ndarray
            MinMax-scaled feature matrix.
        top_n : int, default 20
            Number of top features to consider.
        delta : float, default 0.1
            Perturbation magnitude.

        Returns
        -------
        pd.DataFrame
            Pair interaction results sorted by synergy score.
        """
        self._check_is_fitted()

        inf_batch = self.inference_batch_size or max(self.batch_size // 2, 1024)

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = X.columns.str.strip()
            X = X.select_dtypes(include=[np.number]).values.astype(np.float32)
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        baseline_preds = compute_baseline_predictions(
            self.discriminator_, X, batch_size=inf_batch, verbose=self.verbose
        )

        pair_results = feature_pair_sensitivity(
            model=self.discriminator_,
            data=X,
            baseline_preds=baseline_preds,
            individual_sensitivities=self.sensitivities_,
            feature_names=self.feature_names_,
            batch_size=inf_batch,
            top_n=top_n,
            delta=delta,
            verbose=self.verbose,
        )

        self.pair_results_ = pair_results
        return self._pair_results_to_df(pair_results)

    def save(self, path):
        """Save the fitted GANFS model to disk.

        Saves the generator, discriminator, and metadata (feature names,
        sensitivities, hyperparameters) to the specified directory.

        Parameters
        ----------
        path : str
            Directory path to save the model.
        """
        self._check_is_fitted()
        os.makedirs(path, exist_ok=True)

        # Save Keras models
        self.generator_.save(os.path.join(path, 'generator.keras'))
        self.discriminator_.save(os.path.join(path, 'discriminator.keras'))

        # Save metadata
        metadata = {
            'feature_names': self.feature_names_,
            'sensitivities': self.sensitivities_.tolist(),
            'sorted_indices': self.sorted_indices_.tolist(),
            'n_features': self._n_features,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'perturbation_mode': self.perturbation_mode,
            'perturbation_factors': self.perturbation_factors,
            'generator_hidden_layers': self.generator_hidden_layers,
            'discriminator_hidden_layers': self.discriminator_hidden_layers,
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path):
        """Load a saved GANFS model from disk.

        Parameters
        ----------
        path : str
            Directory path containing the saved model.

        Returns
        -------
        GANFS
            Loaded GANFS instance with restored models and metadata.
        """
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        instance = cls(
            epochs=metadata['epochs'],
            batch_size=metadata['batch_size'],
            learning_rate=metadata['learning_rate'],
            perturbation_mode=metadata['perturbation_mode'],
            perturbation_factors=metadata['perturbation_factors'],
            generator_hidden_layers=metadata.get('generator_hidden_layers', (64, 128)),
            discriminator_hidden_layers=metadata.get('discriminator_hidden_layers', (128, 64)),
        )

        instance.generator_ = tf.keras.models.load_model(
            os.path.join(path, 'generator.keras')
        )
        instance.discriminator_ = tf.keras.models.load_model(
            os.path.join(path, 'discriminator.keras')
        )
        instance.feature_names_ = metadata['feature_names']
        instance.sensitivities_ = np.array(metadata['sensitivities'], dtype=np.float32)
        instance.sorted_indices_ = np.array(metadata['sorted_indices'], dtype=np.int64)
        instance._n_features = metadata['n_features']
        instance.is_fitted_ = True

        return instance

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _prepare_data(self, X, y, label_col):
        """Convert inputs to scaled numpy arrays and extract feature names."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = X.columns.str.strip()

            if y is None:
                # Extract labels from DataFrame
                if label_col in X.columns:
                    y = X[label_col].values
                    X = X.drop(columns=[label_col])
                else:
                    raise ValueError(
                        f"y is None and label column '{label_col}' not found "
                        f"in DataFrame. Either provide y or ensure the DataFrame "
                        f"contains a '{label_col}' column."
                    )

            feature_names = X.columns.tolist()

            # Keep only numeric columns
            X = X.select_dtypes(include=[np.number])
            feature_names = X.columns.tolist()

            # Replace infinities and NaN
            X = X.replace([np.inf, -np.inf, 'Infinity'], 0)
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            X_values = X.values.astype(np.float32)
        else:
            X_values = np.asarray(X, dtype=np.float32)
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]

            # Replace infinities
            X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale to [0, 1]
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X_values)

        return X_scaled, feature_names

    def _build_models(self):
        """Build generator, discriminator, and GAN models."""
        with tf.device(self._device):
            self.generator_ = build_generator(
                self._n_features, 
                self._n_features, 
                hidden_layers=self.generator_hidden_layers
            )
            self.discriminator_ = build_discriminator(
                self._n_features,
                hidden_layers=self.discriminator_hidden_layers
            )

            self.discriminator_.compile(
                loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                metrics=['accuracy']
            )

            # Build GAN (generator → discriminator)
            self.discriminator_.trainable = True
            gan_input = layers.Input(shape=(self._n_features,))
            generated_sample = self.generator_(gan_input)
            gan_output = self.discriminator_(generated_sample)
            self.gan_ = models.Model(gan_input, gan_output)
            self.gan_.compile(
                loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=self.learning_rate)
            )

        if self.verbose:
            logger.info("Generator, Discriminator, and GAN models built.")

    def _train_gan(self, X_scaled):
        """Train the GAN with label smoothing and checkpointing."""
        # Setup checkpointing
        checkpoint = None
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(
                generator=self.generator_,
                discriminator=self.discriminator_,
                gan=self.gan_,
            )

        n_samples = X_scaled.shape[0]
        real_label, fake_label = self.label_smoothing

        if self.verbose:
            logger.info(
                "Starting GAN training: %d epochs, batch_size=%d, device=%s",
                self.epochs, self.batch_size, self._device
            )

        best_d_loss = float('inf')
        wait = 0

        with tf.device(self._device):
            for epoch in range(self.epochs):
                # --- Train Discriminator ---
                idx = np.random.choice(n_samples, self.batch_size, replace=False)
                real_samples = X_scaled[idx]

                noise = np.random.normal(0, 1, (self.batch_size, self._n_features))
                fake_samples = self.generator_.predict(noise, verbose=0)

                real_labels = real_label * np.ones((self.batch_size, 1))
                fake_labels = fake_label * np.ones((self.batch_size, 1))

                d_loss_real = self.discriminator_.train_on_batch(
                    real_samples, real_labels
                )
                d_loss_fake = self.discriminator_.train_on_batch(
                    fake_samples, fake_labels
                )
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # --- Train Generator ---
                noise = np.random.normal(0, 1, (self.batch_size, self._n_features))
                valid_y = np.ones((self.batch_size, 1))
                g_loss = self.gan_.train_on_batch(noise, valid_y)

                # Logging and checkpointing
                d_loss_val = d_loss[0] if isinstance(d_loss, (list, np.ndarray)) else d_loss
                
                if self.verbose and epoch % 20 == 0:
                    d_acc = d_loss[1] if isinstance(d_loss, (list, np.ndarray)) and len(d_loss) > 1 else 0.0
                    logger.info(
                        "Epoch %04d: D_loss=%.4f (acc=%.2f%%), G_loss=%.4f",
                        epoch, d_loss_val, 100 * d_acc, g_loss
                    )
                    if checkpoint:
                        checkpoint.save(file_prefix=checkpoint_prefix)
                
                # Early stopping
                if self.patience is not None:
                    if d_loss_val < best_d_loss - 1e-4:
                        best_d_loss = d_loss_val
                        wait = 0
                    else:
                        wait += 1
                        if wait >= self.patience:
                            if self.verbose:
                                logger.info("Early stopping triggered at epoch %d", epoch)
                            break

        if self.verbose:
            logger.info("GAN training finished.")

        # Re-compile discriminator after training
        self.discriminator_.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

    def _compute_sensitivity(self, X_scaled, batch_size):
        """Run sensitivity analysis and store results."""
        if self.verbose:
            logger.info("Computing baseline predictions...")

        baseline_preds = compute_baseline_predictions(
            self.discriminator_, X_scaled,
            batch_size=batch_size,
            verbose=self.verbose
        )

        if self.verbose:
            logger.info("Running sensitivity analysis...")

        self.sensitivities_ = sensitivity_analysis(
            model=self.discriminator_,
            data=X_scaled,
            baseline_preds=baseline_preds,
            batch_size=batch_size,
            perturbation_mode=self.perturbation_mode,
            perturbation_factors=self.perturbation_factors,
            verbose=self.verbose,
        )

        self.sorted_indices_ = np.argsort(-self.sensitivities_)

    def _check_is_fitted(self):
        """Raise if the model hasn't been fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                "GANFS model is not fitted yet. Call fit(X, y) first."
            )

    @staticmethod
    def _pair_results_to_df(pair_results):
        """Convert pair results list to a DataFrame."""
        return pd.DataFrame({
            'Feature_1': [p['features'][0] for p in pair_results],
            'Feature_2': [p['features'][1] for p in pair_results],
            'Combined_Impact': [p['combined_impact'] for p in pair_results],
            'Synergy_Score': [p['synergy_score'] for p in pair_results],
            'Individual_Sum': [p['individual_sum'] for p in pair_results],
        })
