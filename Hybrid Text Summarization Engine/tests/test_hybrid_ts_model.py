import unittest
from CorpusCondenser.hybrid_ts_model import ExtractiveModel, AbstractiveModel, HybridSummarizationModel

class TestHybridSummarizationModel(unittest.TestCase):

    def setUp(self):
        self.extractive_model = ExtractiveModel(top_n=5)
        self.abstractive_model = AbstractiveModel()
        self.hybrid_model = HybridSummarizationModel(self.extractive_model, self.abstractive_model)

        self.sentences = ["This is a test sentence.", "Another test sentence is here.", "This is the third test sentence.", "Here is the fourth one.", "And finally, the fifth test sentence."]
        self.features = {
            'tfidf': [0.1, 0.2, 0.3, 0.4, 0.5],
            'embeddings': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
            'entities': [('test', 'O')],
            'keywords': ['test'],
            'topics': {'topics': {'test': 0.5}}
        }

    def test_rank_sentences(self):
        ranked_sentences = self.extractive_model.rank_sentences(self.sentences, self.features)
        self.assertEqual(len(ranked_sentences), len(self.sentences))
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in ranked_sentences))

    def test_extractive_summarize(self):
        summary = self.extractive_model.summarize(self.sentences, self.features)
        self.assertEqual(len(summary), self.extractive_model.top_n)

    def test_abstractive_generate_summary(self):
        summary = self.abstractive_model.generate_summary(self.sentences)
        self.assertIsInstance(summary, str)

    def test_hybrid_summarize(self):
        summary = self.hybrid_model.summarize(self.sentences, self.features)
        self.assertIsInstance(summary, str)

    def test_make_human_interpretable(self):
        summary = "this is a test. another test is here."
        human_readable_summary = self.hybrid_model.make_human_interpretable(summary)
        self.assertEqual(human_readable_summary, "This is a test. Another test is here.")

if __name__ == '__main__':
    unittest.main()