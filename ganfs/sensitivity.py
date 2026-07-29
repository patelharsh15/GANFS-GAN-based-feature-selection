"""
Sensitivity analysis for GANFS.

Implements perturbation-based feature importance scoring using the
trained GAN discriminator. Features are ranked by how much the
discriminator's output changes when each feature is perturbed.
"""

import gc
import itertools
import logging
import time

import numpy as np
import tensorflow as tf

logger = logging.getLogger("ganfs")


def compute_baseline_predictions(model, data, batch_size=4096, verbose=False):
    """Compute discriminator predictions on unperturbed data.

    Uses batched inference with automatic memory management to handle
    arbitrarily large datasets without OOM errors.

    Parameters
    ----------
    model : tf.keras.Model
        Trained discriminator model.
    data : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    batch_size : int, default 4096
        Batch size for inference.
    verbose : bool, default False
        If True, print progress information.

    Returns
    -------
    np.ndarray
        Discriminator predictions of shape ``(n_samples, 1)``.
    """
    data = data.astype(np.float32)
    n_samples = len(data)

    # Get output shape from a single sample
    sample_pred = model(data[:1]).numpy()
    baseline_preds = np.empty(
        (n_samples, *sample_pred.shape[1:]), dtype=np.float32
    )

    if verbose:
        logger.info(
            "Computing baseline predictions for %d samples (batch_size=%d)...",
            n_samples, batch_size
        )

    start_time = time.time()
    current_idx = 0
    total_batches = int(np.ceil(n_samples / batch_size))

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        batch = data[start:end]

        pred = model(batch, training=False).numpy()
        batch_len = pred.shape[0]
        baseline_preds[current_idx:current_idx + batch_len] = pred
        current_idx += batch_len

        # Memory management
        del pred
        if batch_idx % 10 == 0:
            gc.collect()

        if verbose and (batch_idx % 100 == 0 or batch_idx == total_batches - 1):
            elapsed = time.time() - start_time
            progress = current_idx / n_samples * 100
            logger.info(
                "  %.1f%% | Batch %d/%d | Elapsed: %.0fs",
                progress, batch_idx + 1, total_batches, elapsed
            )

    if verbose:
        total_time = time.time() - start_time
        logger.info("Baseline predictions complete in %.1fs", total_time)

    return baseline_preds


def sensitivity_analysis(
    model, data, baseline_preds, batch_size=4096,
    perturbation_mode='dynamic',
    perturbation_factors=None,
    verbose=False
):
    """Compute feature sensitivity scores via perturbation analysis.

    For each feature, applies perturbations of varying magnitudes in both
    positive and negative directions and measures how much the
    discriminator's output changes. Features that cause larger changes
    are considered more important.

    In ``'dynamic'`` mode, perturbation magnitudes are scaled based on
    each feature's natural granularity (mean difference between sorted
    unique values).

    Parameters
    ----------
    model : tf.keras.Model
        Trained discriminator model.
    data : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``, MinMax-scaled to [0, 1].
    baseline_preds : np.ndarray
        Unperturbed discriminator predictions from :func:`compute_baseline_predictions`.
    batch_size : int, default 4096
        Batch size for inference.
    perturbation_mode : str, default ``'dynamic'``
        ``'dynamic'`` scales perturbations by feature granularity;
        ``'static'`` uses raw perturbation_factors as absolute deltas.
    perturbation_factors : list[float] or None
        Multipliers (dynamic mode) or absolute deltas (static mode).
        Defaults to ``[0.5, 1.0, 2.0, 5.0, 10.0]``.
    verbose : bool, default False
        If True, print progress information.

    Returns
    -------
    np.ndarray
        Sensitivity scores of shape ``(n_features,)``.
    """
    if perturbation_factors is None:
        perturbation_factors = [0.5, 1.0, 2.0, 5.0, 10.0]

    n_samples, n_features = data.shape
    sensitivities = np.zeros(n_features, dtype=np.float32)
    start_time = time.time()

    # Compute per-feature granularity for dynamic mode
    if perturbation_mode == 'dynamic':
        base_deltas = np.array([
            _calculate_feature_granularity(data[:, i])
            for i in range(n_features)
        ])
        if verbose:
            logger.info("Computed feature granularity for dynamic perturbations.")
    else:
        base_deltas = None

    if verbose:
        logger.info(
            "Starting sensitivity analysis: %d features, %d perturbation levels, mode=%s",
            n_features, len(perturbation_factors), perturbation_mode
        )

    for feat_idx in range(n_features):
        feature_start = time.time()

        # Determine perturbation magnitudes
        if perturbation_mode == 'dynamic':
            deltas = [f * base_deltas[feat_idx] for f in perturbation_factors]
        else:
            deltas = list(perturbation_factors)

        feature_sens = 0.0

        for delta in deltas:
            delta_sens = 0.0

            for direction in [1, -1]:
                total_diff = 0.0

                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_data = data[start:end].copy()
                    original_values = batch_data[:, feat_idx]

                    # Apply perturbation with boundary awareness
                    perturbed = original_values + direction * delta
                    batch_data[:, feat_idx] = np.clip(
                        np.where(
                            (perturbed >= np.min(original_values)) &
                            (perturbed <= np.max(original_values)),
                            perturbed,
                            original_values
                        ), 0, 1
                    )

                    perturbed_preds = model(batch_data, training=False).numpy()
                    total_diff += np.abs(
                        perturbed_preds - baseline_preds[start:end]
                    ).sum()

                mean_diff = total_diff / n_samples
                delta_sens += mean_diff

            # Average across directions
            feature_sens += delta_sens / 2

        sensitivities[feat_idx] = feature_sens / len(deltas)

        # Memory cleanup
        if feat_idx % 5 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

        if verbose:
            elapsed = time.time() - feature_start
            remaining = (time.time() - start_time) * (n_features - feat_idx - 1) / (feat_idx + 1)
            logger.info(
                "  Feature %d/%d: sensitivity=%.6f (%.1fs, ~%.0fs remaining)",
                feat_idx + 1, n_features, sensitivities[feat_idx],
                elapsed, remaining
            )

    total_time = time.time() - start_time
    if verbose:
        logger.info(
            "Sensitivity analysis complete in %.0fm %.0fs",
            total_time // 60, total_time % 60
        )

    return sensitivities


def feature_pair_sensitivity(
    model, data, baseline_preds, individual_sensitivities,
    feature_names, batch_size=4096, top_n=20, delta=0.1,
    verbose=False
):
    """Analyze synergistic interactions between top feature pairs.

    For each pair of top-ranked features, applies a simultaneous
    perturbation and measures whether the combined effect exceeds
    the sum of individual effects (synergy).

    Parameters
    ----------
    model : tf.keras.Model
        Trained discriminator model.
    data : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    baseline_preds : np.ndarray
        Unperturbed discriminator predictions.
    individual_sensitivities : np.ndarray
        Per-feature sensitivity scores from :func:`sensitivity_analysis`.
    feature_names : list[str]
        Feature names corresponding to columns of ``data``.
    batch_size : int, default 4096
        Batch size for inference.
    top_n : int, default 20
        Number of top features to consider for pair analysis.
    delta : float, default 0.1
        Perturbation magnitude.
    verbose : bool, default False
        If True, print progress information.

    Returns
    -------
    list[dict]
        Sorted list of pair results (highest synergy first), each with keys:
        ``'features'``, ``'combined_impact'``, ``'synergy_score'``,
        ``'individual_sum'``.
    """
    n_samples = data.shape[0]
    data = data.astype(np.float32)
    pair_results = []
    start_time = time.time()

    top_indices = np.argsort(-individual_sensitivities)[:top_n]
    pairs = list(itertools.combinations(top_indices, 2))
    total_pairs = len(pairs)

    if verbose:
        logger.info(
            "Starting pair analysis: %d pairs from top %d features",
            total_pairs, top_n
        )

    for pair_idx, (i, j) in enumerate(pairs):
        feat1, feat2 = feature_names[i], feature_names[j]
        total_diff = 0.0

        for batch_start in range(0, n_samples, batch_size):
            end = min(batch_start + batch_size, n_samples)
            batch_data = data[batch_start:end].copy()

            # Apply dual perturbation
            batch_data[:, [i, j]] = np.clip(
                batch_data[:, [i, j]] + delta, 0, 1
            )

            perturbed_preds = model(batch_data, training=False).numpy()
            total_diff += np.abs(
                perturbed_preds - baseline_preds[batch_start:end]
            ).sum()

        combined_sens = total_diff / n_samples
        interaction = combined_sens - (
            individual_sensitivities[i] + individual_sensitivities[j]
        )

        pair_results.append({
            'features': (feat1, feat2),
            'combined_impact': float(combined_sens),
            'synergy_score': float(interaction),
            'individual_sum': float(
                individual_sensitivities[i] + individual_sensitivities[j]
            )
        })

        # Memory cleanup
        if pair_idx % 10 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

        if verbose and (pair_idx % 10 == 0 or pair_idx == total_pairs - 1):
            elapsed = time.time() - start_time
            logger.info(
                "  Pair %d/%d (%s, %s): synergy=%.6f (%.1fs elapsed)",
                pair_idx + 1, total_pairs, feat1, feat2,
                interaction, elapsed
            )

    if verbose:
        total_time = time.time() - start_time
        logger.info("Pair analysis complete in %.0fm %.0fs", total_time // 60, total_time % 60)

    return sorted(pair_results, key=lambda x: x['synergy_score'], reverse=True)


def _calculate_feature_granularity(feature_data):
    """Compute the mean step size between sorted unique values of a feature.

    Used in dynamic perturbation mode to scale perturbations proportionally
    to the natural resolution of each feature.

    Parameters
    ----------
    feature_data : np.ndarray
        1D array of feature values.

    Returns
    -------
    float
        Mean non-zero difference between consecutive sorted values,
        or a fallback minimum if all values are identical.
    """
    sorted_values = np.sort(feature_data)
    diffs = np.diff(sorted_values)
    non_zero_diffs = diffs[diffs > 0]
    if len(non_zero_diffs) == 0:
        return max(1e-3, 0.01 * (np.max(feature_data) - np.min(feature_data)))
    return float(np.mean(non_zero_diffs))
