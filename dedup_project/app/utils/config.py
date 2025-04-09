import argparse
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the deduplication pipeline."""

    # Data paths
    data_path: str

    # Model configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Sampling parameters
    negative_ratio: float = 1.0
    similarity_threshold: float = 0.8
    neighbors: int = 20

    # ANN index parameters
    num_trees: int = 10

    # Training parameters
    test_size: float = 0.2
    random_seed: int = 42

    @classmethod
    def from_args(cls) -> "Config":
        """
        Create Config from command line arguments.

        Returns:
            Config instance with values from command line arguments
        """
        parser = argparse.ArgumentParser(
            description="Supervised Deduplication with Hard Negatives"
        )
        parser.add_argument("--data", required=True, help="Path to the CSV file")
        parser.add_argument(
            "--model",
            default="sentence-transformers/all-MiniLM-L6-v2",
            help="SentenceTransformer model name/path",
        )
        parser.add_argument(
            "--negative-ratio",
            type=float,
            default=1.0,
            help="How many random negative pairs to sample for each positive pair",
        )
        parser.add_argument(
            "--test-size",
            type=float,
            default=0.2,
            help="Fraction of pairs to use as test set",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=42,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--similarity-threshold",
            type=float,
            default=0.80,
            help="Cosine similarity threshold above which an unlabeled pair is considered a hard negative",
        )
        parser.add_argument(
            "--neighbors",
            type=int,
            default=20,
            help="Number of neighbors per doc in ANN search for hard negatives",
        )
        parser.add_argument(
            "--num-trees",
            type=int,
            default=10,
            help="Number of trees to build in Annoy index (higher = better recall, slower build)",
        )

        args = parser.parse_args()

        return cls(
            data_path=args.data,
            model_name=args.model,
            negative_ratio=args.negative_ratio,
            test_size=args.test_size,
            random_seed=args.random_seed,
            similarity_threshold=args.similarity_threshold,
            neighbors=args.neighbors,
            num_trees=args.num_trees,
        )
