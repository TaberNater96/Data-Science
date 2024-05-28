import unittest
from CorpusCondenser.ts_feature_engineering import extract_tfidf_features, extract_sentence_embeddings, extract_named_entities, topic_modeling, extract_keywords, feature_engineering_pipeline

class TestTsFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.sentences = ["This is a test sentence.", "This is another test sentence."]
        self.corpus = " ".join(self.sentences)

    def test_extract_tfidf_features(self):
        tfidf_scores = extract_tfidf_features(self.sentences)
        self.assertIsInstance(tfidf_scores, dict)
        self.assertGreater(len(tfidf_scores), 0)

    def test_extract_sentence_embeddings(self):
        embeddings = extract_sentence_embeddings(self.sentences)
        self.assertEqual(embeddings.shape[0], len(self.sentences))

    def test_extract_named_entities(self):
        entities = extract_named_entities(self.corpus)
        self.assertIsInstance(entities, list)

    def test_topic_modeling(self):
        topics = topic_modeling(self.corpus)
        self.assertIsInstance(topics, dict)
        self.assertIn('distribution', topics)
        self.assertIn('topics', topics)

    def test_extract_keywords(self):
        keywords = extract_keywords(self.corpus)
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

    def test_feature_engineering_pipeline(self):
        features = feature_engineering_pipeline(self.sentences)
        self.assertIsInstance(features, dict)
        self.assertIn('tfidf', features)
        self.assertIn('embeddings', features)
        self.assertIn('entities', features)
        self.assertIn('topics', features)
        self.assertIn('keywords', features)

if __name__ == '__main__':
    unittest.main()