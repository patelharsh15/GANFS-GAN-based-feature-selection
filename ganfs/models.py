"""
GAN model builders for GANFS.

Provides factory functions to construct the Generator and Discriminator
networks used in the GAN-based feature selection pipeline.
"""

import tensorflow as tf

# Use tf.keras to avoid conflicts with standalone keras/JAX installations
layers = tf.keras.layers
models = tf.keras.models


def build_generator(input_dim, output_dim, hidden_layers=(64, 128)):
    """Build the Generator network.

    The generator learns to produce synthetic feature vectors that mimic
    the distribution of real data. The sigmoid activation ensures outputs
    are in [0, 1], matching MinMax-scaled input features.

    Parameters
    ----------
    input_dim : int
        Dimension of the noise input vector.
    output_dim : int
        Dimension of the generated feature vector (should match number of features).
    hidden_layers : tuple[int], default (64, 128)
        Sizes of the hidden Dense layers.

    Returns
    -------
    tf.keras.Model
        Compiled generator model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    return model


def build_discriminator(input_dim, hidden_layers=(128, 64)):
    """Build the Discriminator network.

    The discriminator learns to distinguish real feature vectors from
    synthetic ones. After training, its learned weights encode feature
    importance — features the discriminator relies on most heavily
    are the most discriminative.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    hidden_layers : tuple[int], default (128, 64)
        Sizes of the hidden Dense layers.

    Returns
    -------
    tf.keras.Model
        Compiled discriminator model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
