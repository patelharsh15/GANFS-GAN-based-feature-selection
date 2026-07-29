"""
GAN model builders for GANFS.

Provides factory functions to construct the Generator and Discriminator
networks used in the GAN-based feature selection pipeline.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_generator(input_dim, output_dim):
    """Build the Generator network.

    Architecture: Dense(64, ReLU) → Dense(128, ReLU) → Dense(output_dim, Sigmoid)

    The generator learns to produce synthetic feature vectors that mimic
    the distribution of real data. The sigmoid activation ensures outputs
    are in [0, 1], matching MinMax-scaled input features.

    Parameters
    ----------
    input_dim : int
        Dimension of the noise input vector.
    output_dim : int
        Dimension of the generated feature vector (should match number of features).

    Returns
    -------
    tf.keras.Model
        Compiled generator model.
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    return model


def build_discriminator(input_dim):
    """Build the Discriminator network.

    Architecture: Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid)

    The discriminator learns to distinguish real feature vectors from
    synthetic ones. After training, its learned weights encode feature
    importance — features the discriminator relies on most heavily
    are the most discriminative.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.

    Returns
    -------
    tf.keras.Model
        Compiled discriminator model.
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
