import numpy as np

class DecisionStump:
    """
    Weak ranker mapping X -> {0, 1} based on a single feature and a threshold.
    h(x) = 1 if x[feature_idx] > threshold (for polarity = 1) else 0
    h(x) = 1 if x[feature_idx] <= threshold (for polarity = -1) else 0
    """
    def __init__(self, feature_idx=0, threshold=0.0, polarity=1):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity  # 1 for '>', -1 for '<='

    def predict(self, X):
        """
        Predict labels in {0, 1} for a batch of samples.
        """
        vals = X[:, self.feature_idx]
        if self.polarity == 1:
            return (vals > self.threshold).astype(float)
        else:
            return (vals <= self.threshold).astype(float)

    def __repr__(self):
        op = ">" if self.polarity == 1 else "<="
        return f"Stump(Feature {self.feature_idx} {op} {self.threshold:.4f})"


class RankBoost:
    """
    Pairwise RankBoost algorithm from scratch (Chapter 9, Section 9.4 in FML).
    """
    def __init__(self, n_iterations=50):
        self.n_iterations = n_iterations
        self.stumps = []
        self.alphas = []
        self.Z_all = []  # List of normalization factors Z_t for bound verification
        self.train_loss_history = []
        self.train_error_history = []

    def fit(self, X, pairs):
        """
        Fits RankBoost on the pairwise training data.
        
        Args:
            X: np.ndarray, shape (num_samples, num_features)
            pairs: list of tuples (i, j, y) where:
                   y = +1 if sample j (x'_k) is ranked higher than sample i (x_k)
                   y = -1 if sample j (x'_k) is ranked lower than sample i (x_k)
        """
        M = len(pairs)
        if M == 0:
            raise ValueError("Empty pairwise dataset.")

        # Extract pair indices and labels for vectorization
        idx_left = np.array([p[0] for p in pairs], dtype=int)   # x_k indices
        idx_right = np.array([p[1] for p in pairs], dtype=int)  # x'_k indices
        y_pairs = np.array([p[2] for p in pairs])               # labels y_k

        # Initialize distribution over pairs
        D = np.ones(M) / M

        self.stumps = []
        self.alphas = []
        self.Z_all = []
        self.train_loss_history = []
        self.train_error_history = []

        num_features = X.shape[1]

        # Pre-generate candidate thresholds for each feature to speed up training
        candidates = []
        for d in range(num_features):
            unique_vals = np.sort(np.unique(X[:, d]))
            # Use unique values as threshold candidates
            candidates.append(unique_vals)

        for t in range(self.n_iterations):
            best_stump = None
            best_val = float('inf')  # Minimize eps_minus - eps_plus
            best_eps_plus = 0.0
            best_eps_minus = 0.0
            best_eps_zero = 0.0

            # Find weak ranker h_t that minimizes eps_minus - eps_plus
            for d in range(num_features):
                for theta in candidates[d]:
                    for polarity in [1, -1]:
                        # Create candidate stump
                        stump = DecisionStump(feature_idx=d, threshold=theta, polarity=polarity)
                        H = stump.predict(X)
                        
                        # Compute predictions for all pairs
                        diff = H[idx_right] - H[idx_left]  # h(x'_k) - h(x_k) in {-1, 0, 1}
                        prod = y_pairs * diff              # y_k (h(x'_k) - h(x_k)) in {-1, 0, 1}
                        
                        # Calculate epsilon probabilities
                        eps_plus = np.sum(D[prod == 1])
                        eps_minus = np.sum(D[prod == -1])
                        eps_zero = np.sum(D[prod == 0])
                        
                        val = eps_minus - eps_plus
                        if val < best_val:
                            best_val = val
                            best_stump = stump
                            best_eps_plus = eps_plus
                            best_eps_minus = eps_minus
                            best_eps_zero = eps_zero

            # Safeguard division by zero
            eps = 1e-10
            best_eps_plus = max(best_eps_plus, eps)
            best_eps_minus = max(best_eps_minus, eps)

            # Compute alpha_t and normalization factor Z_t
            alpha_t = 0.5 * np.log(best_eps_plus / best_eps_minus)
            Z_t = best_eps_zero + 2.0 * np.sqrt(best_eps_plus * best_eps_minus)

            # Save step results
            self.stumps.append(best_stump)
            self.alphas.append(alpha_t)
            self.Z_all.append(Z_t)

            # Update distribution
            H_best = best_stump.predict(X)
            diff_best = H_best[idx_right] - H_best[idx_left]
            D = D * np.exp(-alpha_t * y_pairs * diff_best) / Z_t

            # Track metrics for theory verification
            # Compute current scoring function
            scores_curr = self.predict_scores(X)
            diff_all = scores_curr[idx_right] - scores_curr[idx_left]
            
            # Pairwise ranking loss: fraction of pairs misranked or tied
            # (which matches the indicator function 1_{y_k(g(x'_k) - g(x_k)) <= 0})
            misranked = (y_pairs * diff_all <= 0).astype(float)
            emp_error = np.mean(misranked)
            self.train_error_history.append(emp_error)

            # Exponential loss on pairs: F(alpha) = (1/M) * sum(exp(-y_k (g(x'_k) - g(x_k))))
            exp_loss = np.mean(np.exp(-y_pairs * diff_all))
            self.train_loss_history.append(exp_loss)

    def predict_scores(self, X):
        """
        Computes the final ranking score g(x) = sum(alpha_t * h_t(x))
        """
        scores = np.zeros(len(X))
        for alpha, stump in zip(self.alphas, self.stumps):
            scores += alpha * stump.predict(X)
        return scores

    def evaluate_pairs(self, X, pairs):
        """
        Calculates the empirical pairwise ranking error.
        """
        idx_left = np.array([p[0] for p in pairs], dtype=int)
        idx_right = np.array([p[1] for p in pairs], dtype=int)
        y_pairs = np.array([p[2] for p in pairs])

        scores = self.predict_scores(X)
        diff = scores[idx_right] - scores[idx_left]
        misranked = (y_pairs * diff <= 0).astype(float)
        return np.mean(misranked)


class BipartiteRankBoost:
    """
    Bipartite RankBoost algorithm from scratch (Chapter 9, Section 9.5, Figure 9.2 in FML).
    """
    def __init__(self, n_iterations=50):
        self.n_iterations = n_iterations
        self.stumps = []
        self.alphas = []
        self.Z_plus_all = []  # Normalization factors Z^+_t
        self.Z_minus_all = [] # Normalization factors Z^-_t
        self.train_error_history = []  # Bipartite ranking loss history

    def fit(self, X_pos, X_neg):
        """
        Fits Bipartite RankBoost.
        
        Args:
            X_pos: np.ndarray, shape (m, num_features) - Positive instances
            X_neg: np.ndarray, shape (n, num_features) - Negative instances
        """
        m = len(X_pos)
        n = len(X_neg)
        if m == 0 or n == 0:
            raise ValueError("Both positive and negative samples must be provided.")

        # Initialize distributions
        D_pos = np.ones(m) / m
        D_neg = np.ones(n) / n

        self.stumps = []
        self.alphas = []
        self.Z_plus_all = []
        self.Z_minus_all = []
        self.train_error_history = []

        # Combine all data for candidate threshold generation
        X_all = np.vstack([X_pos, X_neg])
        num_features = X_all.shape[1]

        candidates = []
        for d in range(num_features):
            unique_vals = np.sort(np.unique(X_all[:, d]))
            candidates.append(unique_vals)

        for t in range(self.n_iterations):
            best_stump = None
            best_val = float('inf')  # Minimize eps_minus - eps_plus
            best_eps_plus = 0.0
            best_eps_minus = 0.0

            # Find weak ranker h_t that minimizes eps_minus - eps_plus
            # eps_plus = E_{i ~ D_pos} [h(x'_i)]
            # eps_minus = E_{j ~ D_neg} [h(x_j)]
            for d in range(num_features):
                for theta in candidates[d]:
                    for polarity in [1, -1]:
                        stump = DecisionStump(feature_idx=d, threshold=theta, polarity=polarity)
                        
                        H_pos = stump.predict(X_pos)
                        H_neg = stump.predict(X_neg)
                        
                        eps_plus = np.sum(D_pos * H_pos)
                        eps_minus = np.sum(D_neg * H_neg)
                        
                        val = eps_minus - eps_plus
                        if val < best_val:
                            best_val = val
                            best_stump = stump
                            best_eps_plus = eps_plus
                            best_eps_minus = eps_minus

            eps = 1e-10
            best_eps_plus = max(best_eps_plus, eps)
            best_eps_minus = max(best_eps_minus, eps)

            # Compute alpha_t, Z^+_t, Z^-_t
            alpha_t = 0.5 * np.log(best_eps_plus / best_eps_minus)
            sqrt_eps_prod = np.sqrt(best_eps_plus * best_eps_minus)
            Z_plus_t = 1.0 - best_eps_plus + sqrt_eps_prod
            Z_minus_t = 1.0 - best_eps_minus + sqrt_eps_prod

            # Save iteration details
            self.stumps.append(best_stump)
            self.alphas.append(alpha_t)
            self.Z_plus_all.append(Z_plus_t)
            self.Z_minus_all.append(Z_minus_t)

            # Update distributions
            H_pos_best = best_stump.predict(X_pos)
            H_neg_best = best_stump.predict(X_neg)

            D_pos = D_pos * np.exp(-alpha_t * H_pos_best) / Z_plus_t
            D_neg = D_neg * np.exp(+alpha_t * H_neg_best) / Z_minus_t

            # Track Bipartite ranking error (fraction of positive-negative pairs misranked or tied)
            scores_pos = self.predict_scores(X_pos)
            scores_neg = self.predict_scores(X_neg)
            
            # Pairwise error: (1/mn) * sum_{i=1}^m sum_{j=1}^n 1_{g(x'_i) <= g(x_j)}
            # We can compute this vectorially using broadcasting
            pos_grid, neg_grid = np.meshgrid(scores_pos, scores_neg, indexing='ij')
            emp_error = np.mean(pos_grid <= neg_grid)
            self.train_error_history.append(emp_error)

    def predict_scores(self, X):
        """
        Computes the final scoring function g(x) = sum(alpha_t * h_t(x))
        """
        scores = np.zeros(len(X))
        for alpha, stump in zip(self.alphas, self.stumps):
            scores += alpha * stump.predict(X)
        return scores

    def evaluate(self, X_pos, X_neg):
        """
        Calculates the empirical bipartite ranking error: (1/mn) * sum_{i,j} 1_{g(x'_i) <= g(x_j)}
        """
        scores_pos = self.predict_scores(X_pos)
        scores_neg = self.predict_scores(X_neg)
        pos_grid, neg_grid = np.meshgrid(scores_pos, scores_neg, indexing='ij')
        return np.mean(pos_grid <= neg_grid)


class RankingSVM:
    """
    Ranking SVM solved from scratch via Pegasos Primal Subgradient Descent (Section 9.3 in FML).
    Optimizes the pairwise hinge loss:
    min_w 0.5 * ||w||^2 + C * sum_k max(0, 1 - y_k * w^T * (x'_k - x_k))
    """
    def __init__(self, C=1.0, n_epochs=100, learning_rate=0.01, lr_decay=0.99, random_seed=42):
        self.C = C
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.random_seed = random_seed
        self.w = None
        self.train_loss_history = []
        self.train_error_history = []

    def fit(self, X, pairs):
        """
        Fits Ranking SVM on the pairwise training data.
        
        Args:
            X: np.ndarray, shape (num_samples, num_features)
            pairs: list of tuples (i, j, y)
        """
        M = len(pairs)
        if M == 0:
            raise ValueError("Empty pairwise dataset.")

        num_features = X.shape[1]
        
        # Initialize w vector randomly to break symmetry
        rng = np.random.default_rng(self.random_seed)
        self.w = rng.normal(loc=0.0, scale=0.01, size=num_features)

        # Precompute the diff vectors and labels
        # diff_k = x'_k - x_k = X[j] - X[i]
        diff_vectors = np.zeros((M, num_features))
        y_pairs = np.zeros(M)
        for k, (i, j, y) in enumerate(pairs):
            diff_vectors[k] = X[j] - X[i]
            y_pairs[k] = y

        self.train_loss_history = []
        self.train_error_history = []

        lr = self.learning_rate
        lambd = 1.0 / self.C  # Regularization parameter

        for epoch in range(self.n_epochs):
            # Shuffle pairs in each epoch
            indices = np.arange(M)
            rng.shuffle(indices)

            for step, idx in enumerate(indices):
                z = diff_vectors[idx]
                y = y_pairs[idx]
                
                # Check margin condition: y_k * (w^T * z_k)
                margin = y * np.dot(self.w, z)
                
                # Pegasos/SGD subgradient update
                if margin < 1.0:
                    # w = w - lr * (lambd * w - y * z)
                    self.w = (1.0 - lr * lambd) * self.w + lr * y * z
                else:
                    # w = w - lr * lambd * w
                    self.w = (1.0 - lr * lambd) * self.w

            # Apply learning rate decay
            lr *= self.lr_decay

            # Calculate and record metrics at the end of each epoch
            margins = y_pairs * (diff_vectors @ self.w)
            hinge_losses = np.maximum(0.0, 1.0 - margins)
            
            # Primal objective value: 0.5 * ||w||^2 + C * sum(hinge_loss)
            obj_val = 0.5 * np.sum(self.w**2) + self.C * np.sum(hinge_losses)
            self.train_loss_history.append(obj_val)

            # Pairwise ranking error (fraction of misranked pairs)
            misranked = (margins <= 0).astype(float)
            self.train_error_history.append(np.mean(misranked))

    def predict_scores(self, X):
        """
        Computes the final ranking score: g(x) = w^T * x
        """
        if self.w is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.w

    def evaluate_pairs(self, X, pairs):
        """
        Calculates the empirical pairwise ranking error.
        """
        idx_left = np.array([p[0] for p in pairs], dtype=int)
        idx_right = np.array([p[1] for p in pairs], dtype=int)
        y_pairs = np.array([p[2] for p in pairs])

        scores = self.predict_scores(X)
        diff = scores[idx_right] - scores[idx_left]
        misranked = (y_pairs * diff <= 0).astype(float)
        return np.mean(misranked)
