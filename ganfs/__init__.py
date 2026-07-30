"""
GANFS — GAN-based Feature Selection
====================================

A Python library for feature selection using Generative Adversarial Networks.
Train a GAN on your data and use the discriminator's sensitivity to rank
and select the most important features.

Quick Start::

    from ganfs import GANFS

    selector = GANFS(epochs=500, batch_size=4096)
    selector.fit(X, y)
    ranking = selector.get_feature_ranking()
    X_selected = selector.transform(X, k=20)
"""

from ganfs.ganfs import GANFS

__version__ = "0.2.0"
__all__ = ["GANFS", "__version__"]
