#!/usr/bin/env python
# main.py
from app.services.deduplication_service import DeduplicationService
from app.utils.config import Config


def main():
    """Main entry point for the deduplication application."""
    # Parse command line arguments into Config object
    config = Config.from_args()

    # Create deduplication service with configuration
    dedup_service = DeduplicationService(config)

    # Run the full pipeline
    dedup_service.run_pipeline()


if __name__ == "__main__":
    main()
