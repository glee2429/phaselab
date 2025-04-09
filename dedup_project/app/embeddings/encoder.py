from typing import List, Tuple

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """Handles document embedding and nearest neighbor search capabilities."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding encoder.

        Args:
            model_name: Name or path of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.annoy_index = None

    def load_model(self) -> SentenceTransformer:
        """
        Load the SentenceTransformer model.

        Returns:
            Loaded SentenceTransformer model
        """
        self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode_documents(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encodes 'combined_text' in the DataFrame.

        Args:
            df: DataFrame containing 'combined_text' column

        Returns:
            Document embeddings as float32 numpy array
        """
        if self.model is None:
            self.load_model()

        texts = df["combined_text"].tolist()
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True
        )
        return self.embeddings.astype("float32")

    def build_annoy_index(self, num_trees: int = 10) -> AnnoyIndex:
        """
        Creates and builds an Annoy index for fast approximate nearest neighbor search.

        Args:
            num_trees: Number of trees to build in the index (higher = better recall, slower build)

        Returns:
            Built AnnoyIndex
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Encode documents first.")

        emb_dim = self.embeddings.shape[1]
        self.annoy_index = AnnoyIndex(emb_dim, metric="angular")

        for i, emb in enumerate(self.embeddings):
            self.annoy_index.add_item(i, emb)

        self.annoy_index.build(num_trees)
        return self.annoy_index

    def get_nearest_neighbors(
        self, idx: int, k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Get k nearest neighbors for a document.

        Args:
            idx: Index of the document
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (neighbor indices, distances)
        """
        if self.annoy_index is None:
            raise ValueError("Annoy index not built. Call build_annoy_index first.")

        return self.annoy_index.get_nns_by_item(idx, k, include_distances=True)

    @staticmethod
    def angular_to_cosine_similarity(distance: float) -> float:
        """
        Convert angular distance to cosine similarity.

        Args:
            distance: Angular distance from Annoy

        Returns:
            Cosine similarity value [0-1]
        """
        # Angular distance in Annoy = 2 * (1 - cos_sim)
        # => cos_sim = 1.0 - distance/2
        return 1.0 - (distance / 2.0)
