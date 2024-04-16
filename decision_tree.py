class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [len(y[y == i]) for i in range(2)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            feature_index, threshold = self._best_split(X, y)
            if feature_index is not None:
                indices_left = X[:, feature_index] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = feature_index
                node['threshold'] = threshold
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index, best_threshold = None, None
        for feature_index in range(X.shape[1]):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            num_left = [0] * 2
            num_right = num_samples_per_class.copy()
            for i in range(1, len(y)):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(2))
                gini_right = 1.0 - sum((num_right[x] / (len(y) - i)) ** 2 for x in range(2))
                gini = (i * gini_left + (len(y) - i) * gini_right) / len(y)
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return best_feature_index, best_threshold

    def _predict(self, sample, tree):
        if 'predicted_class' in tree:
            return tree['predicted_class']
        else:
            if sample[tree['feature_index']] < tree['threshold']:
                return self._predict(sample, tree['left'])
            else:
                return self._predict(sample, tree['right'])

    def predict(self, X):
        return [self._predict(sample, self.tree) for sample in X]
