import re
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Filter out spurious BeautifulSoup warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class DataProcessor:
    """Handles loading, cleaning, and preprocessing of document data for deduplication."""

    def __init__(self):
        """Initialize the DataProcessor."""
        self.df = None
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}

    def load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """
        Loads the CSV, cleans text, and returns a DataFrame.

        Args:
            csv_path: Path to CSV file containing document data

        Returns:
            DataFrame with cleaned text

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {
            "core_id",
            "processed_title",
            "processed_abstract",
            "labelled_duplicates",
        }
        self.df = pd.read_csv(csv_path)

        # Check columns
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Basic text clean
        self.df["processed_title"] = (
            self.df["processed_title"].fillna("").apply(self._clean_text)
        )
        self.df["processed_abstract"] = (
            self.df["processed_abstract"].fillna("").apply(self._clean_text)
        )
        self.df["combined_text"] = (
            self.df["processed_title"] + " " + self.df["processed_abstract"]
        )

        # Build ID mappings
        self._build_id_mappings()

        return self.df

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Remove HTML tags and extra whitespace.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        # Only apply BeautifulSoup if the text might contain HTML
        if "<" in text and ">" in text:
            text_no_html = BeautifulSoup(text, "lxml").get_text()
        else:
            text_no_html = text
        text_no_html = re.sub(r"\s+", " ", text_no_html).strip()
        return text_no_html

    def _build_id_mappings(self) -> None:
        """
        Creates two dictionaries:
         - id_to_idx: core_id -> row index
         - idx_to_id: row index -> core_id
        """
        self.id_to_idx = {}
        self.idx_to_id = {}
        for i, row in self.df.iterrows():
            cid = row["core_id"]
            self.id_to_idx[cid] = i
            self.idx_to_id[i] = cid

    @staticmethod
    def _parse_label_list(label_str: Optional[str]) -> List[int]:
        """
        Regex-based parse of labelled_duplicates: e.g. "['123','456']" or "['123' '456']"

        Args:
            label_str: String containing list of duplicate IDs

        Returns:
            List of integer IDs
        """
        if not isinstance(label_str, str):
            return []
        matches = re.findall(r"'(\d+)'", label_str)
        return [int(m) for m in matches]

    def build_positive_pairs(self) -> List[Tuple[int, int]]:
        """
        Gathers multi-document clusters into (idA, idB) pairs.

        Returns:
            List of tuple pairs representing duplicate documents
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean_data first.")

        positive_pairs = set()

        for _, row in self.df.iterrows():
            main_id = row["core_id"]
            cluster_ids = [main_id] + self._parse_label_list(row["labelled_duplicates"])
            cluster_ids = [cid for cid in cluster_ids if cid in self.id_to_idx]
            cluster_ids = sorted(set(cluster_ids))
            for a, b in combinations(cluster_ids, 2):
                positive_pairs.add(tuple(sorted([a, b])))

        return list(positive_pairs)
