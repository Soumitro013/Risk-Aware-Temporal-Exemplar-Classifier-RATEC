import numpy as np


class RATEC:
    """
    Risk-Aware Temporal Exemplar Classifier

    Designed for real-time multiclass fault classification.
    """

    def __init__(
        self,
        risk_weights=None,
        rho=0.6,
        epsilon=1e-8
    ):
        self.risk_weights = risk_weights
        self.rho = rho
        self.epsilon = epsilon

        self.medians = None
        self.scales = None
        self.prototypes = None
        self.feature_weights = None
        self.classes = None
        self.temporal_scores = None


    # Robust feature scaling


    def _robust_scale(self, X):
        X_scaled = (
            X - self.medians
        ) / (
            self.scales + self.epsilon
        )

        return X_scaled


    # Training


    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y)

        # Robust feature location
        self.medians = np.median(X, axis=0)

        # Median Absolute Deviation
        mad = np.median(
            np.abs(X - self.medians),
            axis=0
        )

        # Consistent robust scale estimate
        self.scales = 1.4826 * mad

        X_scaled = self._robust_scale(X)

        # Class prototypes
        self.prototypes = {}

        for c in self.classes:

            class_data = X_scaled[y == c]

            self.prototypes[c] = np.median(
                class_data,
                axis=0
            )

        # Equal initial feature weights
        n_features = X.shape[1]

        self.feature_weights = (
            np.ones(n_features) / n_features
        )

        # Default risk weights
        if self.risk_weights is None:

            self.risk_weights = {
                c: 1.0 for c in self.classes
            }

        # Initialize temporal scores
        self.temporal_scores = {
            c: 0.0 for c in self.classes
        }

        return self


    # Distance calculation


    def _distance(self, x, prototype):

        return np.sum(
            self.feature_weights
            * np.abs(x - prototype)
        )


    # Real-time prediction


    def predict_one(self, x):

        x = np.asarray(x, dtype=float)

        x_scaled = self._robust_scale(x)

        scores = {}

        # Instantaneous similarity
        similarities = {}

        for c in self.classes:

            distance = self._distance(
                x_scaled,
                self.prototypes[c]
            )

            similarities[c] = (
                1.0 / (distance + self.epsilon)
            )

        # Temporal update
        for c in self.classes:

            self.temporal_scores[c] = (
                (1 - self.rho)
                * similarities[c]
                +
                self.rho
                * self.temporal_scores[c]
            )

            # Risk-aware score
            scores[c] = (
                self.risk_weights[c]
                * self.temporal_scores[c]
            )

        # Final classification
        predicted_class = max(
            scores,
            key=scores.get
        )

        return predicted_class, scores