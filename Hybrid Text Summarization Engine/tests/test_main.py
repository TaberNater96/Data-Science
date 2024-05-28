import unittest
from unittest.mock import patch
from main import process_pdf

class TestMain(unittest.TestCase):

    def setUp(self):
        self.pdf_path = 'test.pdf'  # A test PDF file path
        self.start_page = 1
        self.end_page = 1

    @patch('main.pdf_text_extractor')
    @patch('main.preprocess_text')
    @patch('main.feature_engineering_pipeline')
    @patch('main.ExtractiveModel')
    @patch('main.AbstractiveModel')
    @patch('main.HybridSummarizationModel')
    def test_process_pdf(self, mock_hybrid_model, mock_abstractive_model, mock_extractive_model, mock_feature_engineering_pipeline, mock_preprocess_text, mock_pdf_text_extractor):
        # Mock the functions and models to avoid actual computation
        mock_pdf_text_extractor.return_value = 'This is a test sentence.'
        mock_preprocess_text.return_value = 'This is a test sentence.'
        mock_feature_engineering_pipeline.return_value = {'tfidf': [1.0], 'embeddings': [1.0], 'entities': ['test'], 'topics': ['test'], 'keywords': ['test']}
        mock_extractive_model.return_value = mock_extractive_model
        mock_abstractive_model.return_value = mock_abstractive_model
        mock_hybrid_model.return_value = mock_hybrid_model
        mock_hybrid_model.summarize.return_value = 'This is a summary.'

        # Call the function to test
        summary = process_pdf(self.pdf_path, self.start_page, self.end_page)

        # Check the function calls and the returned summary
        mock_pdf_text_extractor.assert_called_once_with(self.pdf_path, self.start_page, self.end_page)
        mock_preprocess_text.assert_called_once_with(mock_pdf_text_extractor.return_value)
        mock_feature_engineering_pipeline.assert_called_once_with(['This is a test sentence.'])
        mock_extractive_model.assert_called_once_with(top_n=5)
        mock_abstractive_model.assert_called_once_with(model_name='facebook/bart-large-cnn', framework='pt')
        mock_hybrid_model.assert_called_once_with(mock_extractive_model, mock_abstractive_model)
        mock_hybrid_model.summarize.assert_called_once_with(['This is a test sentence.'], mock_feature_engineering_pipeline.return_value)
        self.assertEqual(summary, 'This is a summary.')

if __name__ == '__main__':
    unittest.main()