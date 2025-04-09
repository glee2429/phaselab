import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split


class DuplicateClassifier:
    """
    Handles document pair feature creation, model training,
    and evaluation for document deduplication.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the duplicate classifier.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.X = None
        self.y = None

        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def build_features_for_pair(
        self, idA: int, idB: int, embeddings: np.ndarray, id_to_idx: Dict[int, int]
    ) -> np.ndarray:
        """
        Build feature vector for a document pair.

        Args:
            idA: First document ID
            idB: Second document ID
            embeddings: Document embeddings
            id_to_idx: Mapping from document ID to embedding index

        Returns:
            Feature vector [eA, eB, |eA - eB|, eA*eB]
        """
        iA = id_to_idx[idA]
        iB = id_to_idx[idB]
        eA = embeddings[iA]
        eB = embeddings[iB]
        diff = np.abs(eA - eB)
        mult = eA * eB
        return np.concatenate([eA, eB, diff, mult])

    def sample_random_negatives(
        self,
        positive_pairs: List[Tuple[int, int]],
        id_to_idx: Dict[int, int],
        negative_ratio: float = 1.0,
    ) -> Set[Tuple[int, int]]:
        """
        Sample random negative pairs.

        Args:
            positive_pairs: List of positive (duplicate) pairs
            id_to_idx: Mapping from document ID to embedding index
            negative_ratio: Ratio of negative to positive pairs

        Returns:
            Set of randomly sampled negative pairs
        """
        pos_set = set(positive_pairs)
        num_neg = int(len(positive_pairs) * negative_ratio)

        all_ids = list(id_to_idx.keys())
        random_negatives = set()
        while len(random_negatives) < num_neg:
            idA, idB = random.sample(all_ids, 2)
            pair = tuple(sorted([idA, idB]))
            if pair not in pos_set:
                random_negatives.add(pair)

        return random_negatives

    def gather_hard_negatives(
        self,
        embeddings: np.ndarray,
        annoy_index,
        idx_to_id: Dict[int, int],
        positive_pairs: List[Tuple[int, int]],
        neighbors: int = 20,
        similarity_threshold: float = 0.80,
    ) -> Set[Tuple[int, int]]:
        """
        Find hard negative pairs using nearest neighbor search.

        Args:
            embeddings: Document embeddings
            annoy_index: Built Annoy index
            idx_to_id: Mapping from embedding index to document ID
            positive_pairs: List of positive (duplicate) pairs
            neighbors: Number of neighbors to search per document
            similarity_threshold: Similarity threshold for hard negatives

        Returns:
            Set of hard negative pairs
        """
        from ..embeddings.encoder import EmbeddingEncoder

        pos_set = set(positive_pairs)
        hard_negatives = set()
        n_docs = embeddings.shape[0]

        for i in range(n_docs):
            nns, dists = annoy_index.get_nns_by_item(
                i, neighbors, include_distances=True
            )
            main_doc_id = idx_to_id[i]
            for j, dist in zip(nns, dists):
                if j == i:
                    continue
                cos_sim = EmbeddingEncoder.angular_to_cosine_similarity(dist)
                if cos_sim >= similarity_threshold:
                    neighbor_doc_id = idx_to_id[j]
                    pair = tuple(sorted([main_doc_id, neighbor_doc_id]))
                    if pair not in pos_set:
                        hard_negatives.add(pair)

        return hard_negatives

    def build_dataset(
        self,
        positive_pairs: List[Tuple[int, int]],
        negatives: Set[Tuple[int, int]],
        embeddings: np.ndarray,
        id_to_idx: Dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix and labels from positive and negative pairs.

        Args:
            positive_pairs: List of positive (duplicate) pairs
            negatives: Set of negative (non-duplicate) pairs
            embeddings: Document embeddings
            id_to_idx: Mapping from document ID to embedding index

        Returns:
            Tuple of (feature matrix X, labels y)
        """
        pairs_labeled = [(p[0], p[1], 1) for p in positive_pairs] + [
            (p[0], p[1], 0) for p in negatives
        ]
        random.shuffle(pairs_labeled)

        X, y = [], []
        for idA, idB, label in pairs_labeled:
            feats = self.build_features_for_pair(idA, idB, embeddings, id_to_idx)
            X.append(feats)
            y.append(label)

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int32)
        return self.X, self.y

    def train_and_evaluate(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        test_size: float = 0.2,
    ) -> LogisticRegression:
        """
        Splits data into train/test, trains logistic regression,
        prints metrics, and plots ROC.

        Args:
            X: Feature matrix (optional, uses self.X if None)
            y: Labels (optional, uses self.y if None)
            test_size: Fraction of data to use for testing

        Returns:
            Trained logistic regression model
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        if X is None or y is None:
            raise ValueError("No dataset available. Build dataset first.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        print("====================================")
        print("Evaluation on test set (with Hard Negatives):")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 score:  {f1:.4f}")
        print("====================================")

        # Plot ROC
        y_scores = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"LogReg (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic with Hard Negatives")
        plt.legend()
        plt.show()

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature matrix

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")

        return self.model.predict_proba(X)
