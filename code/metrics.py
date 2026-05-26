import numpy as np

def compute_pairwise_ranking_error(scores, pairs):
    """
    Computes the pairwise ranking error: the fraction of pairs for which
    the predicted scores do not match the true preference ordering.
    
    R(g) = (1/M) * sum_{k=1}^M I( y_k * (g(x'_k) - g(x_k)) <= 0 )
    """
    idx_left = np.array([p[0] for p in pairs], dtype=int)
    idx_right = np.array([p[1] for p in pairs], dtype=int)
    y_pairs = np.array([p[2] for p in pairs])
    
    diff = scores[idx_right] - scores[idx_left]
    misranked = (y_pairs * diff <= 0).astype(float)
    return np.mean(misranked)

def compute_bipartite_ranking_error(scores_pos, scores_neg):
    """
    Computes the bipartite ranking error (fraction of positive-negative pairs
    where the positive sample is ranked lower than or equal to the negative sample).
    
    R(g) = (1 / (m * n)) * sum_{i=1}^m sum_{j=1}^n I( g(x'_i) <= g(x_j) )
    """
    # Using broadcasting to compare all positive and negative scores
    # pos_grid shape: (m, 1), neg_grid shape: (1, n)
    pos_grid = scores_pos[:, np.newaxis]
    neg_grid = scores_neg[np.newaxis, :]
    
    misranked = (pos_grid <= neg_grid).astype(float)
    return np.mean(misranked)

def compute_roc_curve(scores_pos, scores_neg):
    """
    Computes the ROC curve from scratch for a set of positive and negative scores.
    Returns:
        fpr: np.ndarray, false positive rates
        tpr: np.ndarray, true positive rates
        thresholds: np.ndarray, threshold values used
    """
    m = len(scores_pos)
    n = len(scores_neg)
    
    # Combine scores and create labels (1 for positive, 0 for negative)
    all_scores = np.concatenate([scores_pos, scores_neg])
    all_labels = np.concatenate([np.ones(m), np.zeros(n)])
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_scores = all_scores[sorted_indices]
    sorted_labels = all_labels[sorted_indices]
    
    # Add an extreme threshold at the beginning
    thresholds = np.concatenate([[sorted_scores[0] + 1.0], sorted_scores])
    
    tpr = []
    fpr = []
    
    # Running counts of True Positives and False Positives
    tp = 0
    fp = 0
    
    tpr.append(0.0)
    fpr.append(0.0)
    
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / m)
        fpr.append(fp / n)
        
    return np.array(fpr), np.array(tpr), thresholds

def compute_auc_from_scores(scores_pos, scores_neg):
    """
    Computes the Area Under the ROC Curve (AUC) from positive and negative scores.
    By standard definition (Mann-Whitney U statistic):
    AUC = (1 / (m * n)) * sum_{i=1}^m sum_{j=1}^n [ I(g(x'_i) > g(x_j)) + 0.5 * I(g(x'_i) == g(x_j)) ]
    """
    pos_grid = scores_pos[:, np.newaxis]
    neg_grid = scores_neg[np.newaxis, :]
    
    # Compute indicator for strictly greater and equal
    greater = (pos_grid > neg_grid).astype(float)
    equal = (pos_grid == neg_grid).astype(float)
    
    return np.mean(greater + 0.5 * equal)

def compute_auc_book(scores_pos, scores_neg):
    """
    Computes AUC according to the specific textbook definition (Chapter 9, page 225, Figure 9.3):
    AUC = (1 / (m * n)) * sum_{i=1}^m sum_{j=1}^n I(g(x'_i) >= g(x_j))
    
    Under this definition: R(h) = 1 - AUC(h) holds exactly, since:
    R(h) = (1 / (m * n)) * sum_{i=1}^m sum_{j=1}^n I(g(x'_i) <= g(x_j))
    but note that ties make them sum to slightly more than 1, so we define
    strict AUC complement to verify the relation.
    """
    pos_grid = scores_pos[:, np.newaxis]
    neg_grid = scores_neg[np.newaxis, :]
    
    correctly_ranked = (pos_grid >= neg_grid).astype(float)
    return np.mean(correctly_ranked)

def compute_ndcg(scores, true_scores, k=None):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) for a set of items.
    NDCG = DCG / IDCG
    
    Args:
        scores: np.ndarray, predicted scores of the items.
        true_scores: np.ndarray, true relevance/scores of the items.
        k: int, optional, cut-off rank.
    """
    N = len(scores)
    if k is None or k > N:
        k = N
        
    # Sort items by predicted scores descending
    pred_sort_idx = np.argsort(scores)[::-1]
    sorted_true_relevance = true_scores[pred_sort_idx]
    
    # Sort items by true scores descending to get ideal ranking
    ideal_sort_idx = np.argsort(true_scores)[::-1]
    ideal_true_relevance = true_scores[ideal_sort_idx]
    
    # Compute DCG: sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    # Using the standard information retrieval formula
    discounts = np.log2(np.arange(1, k + 1) + 1)
    
    dcg = np.sum((2**sorted_true_relevance[:k] - 1) / discounts)
    idcg = np.sum((2**ideal_true_relevance[:k] - 1) / discounts)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg
