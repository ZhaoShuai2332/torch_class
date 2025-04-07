import numpy as np

class DecisionTree():
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Stop criteria
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[0]

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.random.choice(unique_classes)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent_y, left_y, right_y):
        parent_entropy = self._entropy(parent_y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)

        return parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)

    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    
    def _predict_sample(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_subtree, right_subtree = tree
        
        if sample[feature] <= threshold:
            return self._predict_sample(sample, left_subtree)
        else:
            return self._predict_sample(sample, right_subtree)
        
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def __str__(self):
        return f"DecisionTree(max_depth={self.max_depth})"
    
    def __repr__(self):
        return self.__str__()
    
    def get_params(self, deep=True):
        return {"max_depth": self.max_depth}
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_tree(self):
        return self.tree
    
    def get_depth(self):
        return self._get_depth(self.tree)
    
    def _get_depth(self, tree):
        if not isinstance(tree, tuple):
            return 1
        _, _, left_subtree, right_subtree = tree
        return 1 + max(self._get_depth(left_subtree), self._get_depth(right_subtree))
    
    def get_n_leaves(self):
        return self._get_n_leaves(self.tree)
    
    def save(self, *args, **kwargs):
        super(ModelName, self).save(*args, **kwargs) # Call the real save() method
    
