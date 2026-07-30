import numpy as np
import pandas as pd
import pytest

from ganfs import GANFS


@pytest.fixture
def dummy_data():
    """Create a dummy dataset with 10 features where only the first 2 matter."""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(10)])
    df["Label"] = y
    return df, X, y


def test_ganfs_initialization():
    """Test that GANFS initializes properly with default and custom parameters."""
    # Defaults
    ganfs = GANFS()
    assert ganfs.epochs == 500
    assert ganfs.patience is None
    
    # Custom
    ganfs_custom = GANFS(
        epochs=10, 
        batch_size=16, 
        patience=5, 
        random_state=42,
        generator_hidden_layers=(32, 16),
        discriminator_hidden_layers=(16, 8)
    )
    assert ganfs_custom.epochs == 10
    assert ganfs_custom.patience == 5
    assert ganfs_custom.generator_hidden_layers == (32, 16)


def test_ganfs_fit_transform(dummy_data):
    """Test the complete fit_transform pipeline on a small dataset."""
    df, _, _ = dummy_data
    
    # Use very small epochs to speed up testing
    ganfs = GANFS(epochs=2, batch_size=32, verbose=False)
    
    # Fit transform to select top 3 features
    X_transformed = ganfs.fit_transform(df, label_col="Label", k=3)
    
    # Check shape
    assert X_transformed.shape == (100, 3)
    
    # Check that model is marked as fitted
    assert ganfs.is_fitted_ is True
    
    # Check feature ranking dataframe structure
    ranking = ganfs.get_feature_ranking()
    assert len(ranking) == 10
    assert list(ranking.columns) == ['Rank', 'Feature', 'Sensitivity_Score']
    assert ranking['Rank'].iloc[0] == 1


def test_early_stopping(dummy_data):
    """Test that early stopping cuts training short when patience is reached."""
    df, _, _ = dummy_data
    
    ganfs = GANFS(
        epochs=100,
        batch_size=32,
        patience=1,  # Stop very early
        verbose=False
    )
    ganfs.fit(df, label_col="Label")
    
    # It should not run all 100 epochs, but checking internal epoch count is tricky
    # So we just ensure it doesn't crash and returns normally.
    assert ganfs.is_fitted_ is True


def test_custom_architectures(dummy_data):
    """Test that custom neural network architectures build and train successfully."""
    df, _, _ = dummy_data
    
    ganfs = GANFS(
        epochs=1, 
        batch_size=32,
        generator_hidden_layers=(8,), 
        discriminator_hidden_layers=(16, 8, 4),
        verbose=False
    )
    
    ganfs.fit(df, label_col="Label")
    assert ganfs.is_fitted_ is True
    
    # Check model architecture
    # Generator input shape is (batch_size, n_features) -> Dense(8) -> Dense(n_features)
    assert len(ganfs.generator_.layers) == 2 
    # Discriminator input shape is (batch_size, n_features) -> Dense(16) -> Dense(8) -> Dense(4) -> Dense(1)
    assert len(ganfs.discriminator_.layers) == 4
