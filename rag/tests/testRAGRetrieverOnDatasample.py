# tests/test_real_data_integration.py
import unittest
import os
import shutil
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Import your classes
from rag.database.RAGRetriever import RAGRetrieverGlobal
from rag.entity.dataDB import initDBGet, loadEmbeddingsDBGet, findTopNearestDBGet


# We can keep one setUpClass for all tests in this file to load the model only once.
def setUpModule():
    """Called once before all tests in this module."""
    print("\n--- Loading SentenceTransformer model (can take a moment) ---")
    TestRealDataIntegration.model = SentenceTransformer('all-MiniLM-L6-v2')
    print("--- Model loaded successfully ---")


def tearDownModule():
    """Called once after all tests in this module."""
    print("\n--- All integration tests finished ---")
    TestRealDataIntegration.model = None


class TestRealDataIntegration(unittest.TestCase):
    """
    Integration test checking the full pipeline with "real" data.
    This base class will be inherited by language-specific test classes.
    """
    model = None  # Will be populated by setUpModule

    def setUp(self):
        """Creates temporary directories for DB and data files before each test."""
        # Use a unique name for each test class to avoid conflicts if run in parallel
        self.temp_dir = f"./integration_test_temp_{self.__class__.__name__}"
        self.data_path = os.path.join(self.temp_dir, "data")
        self.db_path = os.path.join(self.temp_dir, "db")
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.db_path, exist_ok=True)

    def tearDown(self):
        """Removes temporary directories and files after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_jsonl_file(self, data, filepath):
        """Helper function to write data to a .jsonl file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class TestEnglishDataIntegration(TestRealDataIntegration):
    """Tests the pipeline with English language data."""

    def setUp(self):
        super().setUp()
        # --- Create "toy" corpus data in English ---
        corpus_data = [
            {"doc_id": "doc_finance_01", "abstract": [
                "Stock market analysis requires understanding of financial trends.",
                "Investment in volatile assets carries significant risk.",
                "A diversified portfolio helps to mitigate potential losses."
            ]},
            {"doc_id": "doc_biology_02", "abstract": [
                "Cellular respiration is a key process for all living organisms.",
                "Photosynthesis converts light energy into chemical energy.",
                "DNA contains the genetic instructions for development."
            ]}
        ]

        # --- Create "toy" queries in English ---
        queries_data = [
            {"id": "q_finance_1", "claim": "How can one reduce investment risks?"},
            {"id": "q_biology_2", "claim": "What is the function of DNA in a cell?"}
        ]

        # --- Save data to .jsonl files ---
        self.corpus_file = os.path.join(self.data_path, 'corpus_en.jsonl')
        self._create_jsonl_file(corpus_data, self.corpus_file)

        self.queries_file = os.path.join(self.data_path, 'queries_en.jsonl')
        self._create_jsonl_file(queries_data, self.queries_file)

    def test_full_pipeline_english(self):
        """Tests the full cycle: load, index, and search on English data."""
        # 1. ARRANGE: Load data and prepare sentences
        corpus_df = pd.read_json(self.corpus_file, lines=True)
        queries_df = pd.read_json(self.queries_file, lines=True)

        all_sentences = sorted(list(set(sent for sublist in corpus_df['abstract'] for sent in sublist)))
        self.assertEqual(len(all_sentences), 6)

        # 2. ACT: Create embeddings and build the index
        retriever = RAGRetrieverGlobal(initDBGet(db_path=self.db_path))
        sentence_embeddings = self.model.encode(all_sentences, convert_to_numpy=True)
        retriever.loadEmbeddingsDB(
            loadEmbeddingsDBGet(sentence_embeddings=sentence_embeddings, sentences=all_sentences))

        self.assertTrue(retriever._is_ready)

        # 3. ACT: Search for the queries
        queries = queries_df['claim'].tolist()
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)

        # --- Search for the finance query ---
        search_data_finance = findTopNearestDBGet(query_embedding=query_embeddings[0], k=3)
        results_finance = retriever.findTopNearestDB(search_data_finance)

        # --- Search for the biology query ---
        search_data_biology = findTopNearestDBGet(query_embedding=query_embeddings[1], k=3)
        results_biology = retriever.findTopNearestDB(search_data_biology)

        # 4. ASSERT: Check the relevance of the results
        self.assertFalse(results_finance.error.isError)
        self.assertEqual(len(results_finance.topNearest), 3)

        # Check that finance query results are all finance-related sentences
        for sentence in results_finance.topNearest:
            self.assertIn(sentence.lower(), [
                "stock market analysis requires understanding of financial trends.",
                "investment in volatile assets carries significant risk.",
                "a diversified portfolio helps to mitigate potential losses."
            ])
            self.assertNotIn("dna", sentence.lower())  # Make sure no biology sentences crept in

        self.assertFalse(results_biology.error.isError)
        self.assertEqual(len(results_biology.topNearest), 3)

        # Check that biology query results are all biology-related sentences
        for sentence in results_biology.topNearest:
            self.assertIn(sentence.lower(), [
                "cellular respiration is a key process for all living organisms.",
                "photosynthesis converts light energy into chemical energy.",
                "dna contains the genetic instructions for development."
            ])
            self.assertNotIn("risk", sentence.lower())  # Make sure no finance sentences crept in


if __name__ == '__main__':
    unittest.main()