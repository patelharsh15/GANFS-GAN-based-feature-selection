import pandas as pd
import numpy as np
from ganfs import GANFS

def run_test():
    print("Generating dummy data for testing...")
    # Create 500 samples with 10 features
    np.random.seed(42)
    X = np.random.rand(500, 10)
    
    # Create a target label that heavily depends on feature 0 and 1, but ignores the rest
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(10)])
    df["Label"] = y

    print("\nInitializing GANFS...")
    # Testing new features: patience=5 and random_state=42
    selector = GANFS(
        epochs=50, 
        batch_size=32, 
        patience=5, 
        random_state=42, 
        verbose=True
    )

    print("\nFitting GANFS...")
    # This should trigger the new Early Stopping logic and print progress
    selector.fit(df, label_col="Label")

    print("\n--- Feature Ranking ---")
    print(selector.get_feature_ranking())
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_test()
