from ..data.processor import DataProcessor
from ..embeddings.encoder import EmbeddingEncoder
from ..models.classifier import DuplicateClassifier
from ..utils.config import Config


class DeduplicationService:
    """Service that orchestrates the entire deduplication pipeline."""

    def __init__(self, config: Config):
        """
        Initialize deduplication service with configuration.

        Args:
            config: Configuration for the deduplication process
        """
        self.config = config
        self.data_processor = DataProcessor()
        self.embedding_encoder = EmbeddingEncoder(model_name=config.model_name)
        self.classifier = DuplicateClassifier(random_seed=config.random_seed)

        # State variables
        self.df = None
        self.embeddings = None
        self.positive_pairs = None
        self.hard_negatives = None
        self.random_negatives = None
        self.combined_negatives = None
        self.X = None
        self.y = None

    def load_and_process_data(self) -> None:
        """Load and process the input data."""
        print(f"Loading and processing data from {self.config.data_path}")
        self.df = self.data_processor.load_and_clean_data(self.config.data_path)
        print(f"Data shape: {self.df.shape}")

        # Build positive pairs from multi-doc clusters
        self.positive_pairs = self.data_processor.build_positive_pairs()
        print(
            f"Total positive (duplicate) pairs from clusters: {len(self.positive_pairs)}"
        )

    def compute_embeddings(self) -> None:
        """Compute document embeddings and build ANN index."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data first.")

        print(f"Computing embeddings using model: {self.config.model_name}")
        self.embeddings = self.embedding_encoder.encode_documents(self.df)
        emb_dim = self.embeddings.shape[1]

        print(f"Building Annoy index with {self.config.num_trees} trees")
        annoy_index = self.embedding_encoder.build_annoy_index(
            num_trees=self.config.num_trees
        )
        print(
            f"Annoy index built with {self.config.num_trees} trees for dimension={emb_dim}"
        )

        return annoy_index

    def generate_negative_pairs(self, annoy_index) -> None:
        """Generate both hard and random negative pairs."""
        if self.embeddings is None or self.positive_pairs is None:
            raise ValueError("Embeddings or positive pairs not available.")

        # Hard negatives with ANN search
        print(
            f"Generating hard negatives with similarity threshold {self.config.similarity_threshold}"
        )
        self.hard_negatives = self.classifier.gather_hard_negatives(
            embeddings=self.embeddings,
            annoy_index=annoy_index,
            idx_to_id=self.data_processor.idx_to_id,
            positive_pairs=self.positive_pairs,
            neighbors=self.config.neighbors,
            similarity_threshold=self.config.similarity_threshold,
        )
        print(f"Hard negatives found: {len(self.hard_negatives)}")

        # Random negatives
        print(f"Sampling random negatives with ratio {self.config.negative_ratio}")
        self.random_negatives = self.classifier.sample_random_negatives(
            self.positive_pairs,
            self.data_processor.id_to_idx,
            negative_ratio=self.config.negative_ratio,
        )
        print(f"Random negatives: {len(self.random_negatives)}")

        # Combine both types of negatives
        self.combined_negatives = self.random_negatives.union(self.hard_negatives)
        print(f"Total negatives: {len(self.combined_negatives)}")

    def build_and_train_model(self) -> None:
        """Build dataset and train the classifier."""
        if (
            self.embeddings is None
            or self.positive_pairs is None
            or self.combined_negatives is None
        ):
            raise ValueError("Required components not available.")

        print("Building dataset from positive and negative pairs")
        self.X, self.y = self.classifier.build_dataset(
            self.positive_pairs,
            self.combined_negatives,
            self.embeddings,
            self.data_processor.id_to_idx,
        )
        print(f"Feature matrix shape: {self.X.shape}, label shape: {self.y.shape}")

        print(f"Training and evaluating model with test_size={self.config.test_size}")
        model = self.classifier.train_and_evaluate(test_size=self.config.test_size)
        return model

    def run_pipeline(self) -> None:
        """Run the complete deduplication pipeline."""
        # 1. Load and process data
        self.load_and_process_data()

        # 2. Compute embeddings and build ANN index
        annoy_index = self.compute_embeddings()

        # 3. Generate negative pairs (hard + random)
        self.generate_negative_pairs(annoy_index)

        # 4. Build features and train model
        self.build_and_train_model()

        print("Deduplication pipeline completed successfully!")
