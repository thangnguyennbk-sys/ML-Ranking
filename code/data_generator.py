import numpy as np

def generate_pairwise_data(num_samples=120, num_features=5, noise=0.1, random_seed=42):
    """
    Generates synthetic data for pairwise ranking.
    
    1. Generates 'num_samples' instances with 'num_features' features.
    2. Computes a true score function with some non-linear component to make it interesting.
    3. Samples pairs of instances and assigns pairwise labels y_i in {-1, 1}:
       y_i = +1 if score(x'_i) > score(x_i)
       y_i = -1 if score(x'_i) < score(x_i)
    
    Returns:
        X: np.ndarray, shape (num_samples, num_features) - Feature matrix of individual samples.
        scores: np.ndarray, shape (num_samples,) - True score of each sample.
        pairs: list of tuples (x_idx, x_prime_idx, y) - Pairwise dataset.
    """
    rng = np.random.default_rng(random_seed)
    
    # 1. Generate feature vectors from normal distribution
    X = rng.normal(loc=0.0, scale=1.0, size=(num_samples, num_features))
    
    # 2. True weight vector for score computation
    # Make the first 3 features important, others noise-like
    true_w = np.array([1.5, -1.0, 0.8] + [0.0] * (num_features - 3))
    
    # Compute scores: linear score + non-linear term (interaction & sine) + noise
    linear_score = X @ true_w
    non_linear_score = 0.5 * np.sin(X[:, 0]) + 0.3 * (X[:, 1] * X[:, 2])
    scores = linear_score + non_linear_score + rng.normal(loc=0.0, scale=noise, size=num_samples)
    
    # 3. Create pairwise dataset
    # We sample a subset of all possible pairs to avoid O(N^2) explosion if N is large,
    # but for N=120, N*(N-1)/2 = 7140 pairs, which is manageable.
    pairs = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            # Check difference in scores
            diff = scores[j] - scores[i]
            if np.abs(diff) < 0.05:  # Skip pairs with scores too close to avoid ambiguity
                continue
            
            y = 1.0 if diff > 0 else -1.0
            # Pair: (x_i, x_j, y) means order is (x_i, x_prime=x_j, y)
            pairs.append((i, j, y))
            
    return X, scores, pairs

def generate_bipartite_data(num_pos=60, num_neg=60, num_features=2, distance=1.0, random_seed=42):
    """
    Generates synthetic data for bipartite ranking.
    Positive instances and negative instances are drawn from different distributions.
    
    Positive: N( [distance/2, ..., distance/2], I )
    Negative: N( [-distance/2, ..., -distance/2], I )
    
    Returns:
        X_pos: np.ndarray, shape (num_pos, num_features)
        X_neg: np.ndarray, shape (num_neg, num_features)
    """
    rng = np.random.default_rng(random_seed)
    
    mean_pos = np.ones(num_features) * (distance / 2.0)
    mean_neg = -np.ones(num_features) * (distance / 2.0)
    
    X_pos = rng.multivariate_normal(mean_pos, np.eye(num_features), size=num_pos)
    X_neg = rng.multivariate_normal(mean_neg, np.eye(num_features), size=num_neg)
    
    return X_pos, X_neg
