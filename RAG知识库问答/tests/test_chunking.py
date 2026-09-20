# -*- coding: utf-8 -*-
"""Unit tests for chunking strategies."""
import unittest

from src.loading.pdf_loader import PageText
from src.chunking.registry import get_strategy, list_strategies


class TestFixedOverlap(unittest.TestCase):
    def setUp(self):
        self.pages = [
            PageText(text="A" * 1000, page_num=1, doc_name="test"),
        ]
        self.config = {
            "fixed_overlap": {"chunk_size": 512, "chunk_overlap": 64},
        }

    def test_chunk_count(self):
        strategy = get_strategy("fixed_overlap")
        chunks = strategy.chunk(self.pages, self.config)
        self.assertGreater(len(chunks), 1)

    def test_chunk_ids_unique(self):
        strategy = get_strategy("fixed_overlap")
        chunks = strategy.chunk(self.pages, self.config)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_strategy_name(self):
        chunks = get_strategy("fixed_overlap").chunk(self.pages, self.config)
        self.assertTrue(all(c.strategy_name == "fixed_overlap" for c in chunks))


class TestSemanticParagraph(unittest.TestCase):
    def setUp(self):
        text = "First paragraph. " * 20 + "\n\n" + "Second paragraph. " * 20
        self.pages = [PageText(text=text, page_num=1, doc_name="test")]
        self.config = {"semantic_paragraph": {"max_chunk_size": 512}}

    def test_produces_chunks(self):
        strategy = get_strategy("semantic_paragraph")
        chunks = strategy.chunk(self.pages, self.config)
        self.assertGreater(len(chunks), 0)

    def test_no_empty_chunks(self):
        strategy = get_strategy("semantic_paragraph")
        chunks = strategy.chunk(self.pages, self.config)
        self.assertTrue(all(c.text for c in chunks))


class TestRegistry(unittest.TestCase):
    def test_list_strategies(self):
        strategies = list_strategies()
        self.assertIn("fixed_overlap", strategies)
        self.assertIn("semantic_paragraph", strategies)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("nonexistent_strategy")


if __name__ == "__main__":
    unittest.main()
