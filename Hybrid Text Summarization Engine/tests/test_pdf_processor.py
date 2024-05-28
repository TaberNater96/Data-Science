import unittest
import os
import PyPDF2
from CorpusCondenser.pdf_processor import pdf_text_extractor, save_text_to_file

class TestPdfProcessor(unittest.TestCase):

    def setUp(self):
        self.filepath = 'test.pdf'  
        self.output_filepath = 'output.txt'  

    def test_pdf_text_extractor(self):
        text = pdf_text_extractor(self.filepath)
        with open(self.filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            expected_text = ""
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                expected_text += page.extractText()
        self.assertEqual(text, expected_text)

    def test_save_text_to_file(self):
        text = "This is a test text."
        save_text_to_file(text, self.output_filepath)
        with open(self.output_filepath, 'r', encoding='utf-8') as file:
            saved_text = file.read()
        self.assertEqual(text, saved_text)

    def tearDown(self):
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)

if __name__ == '__main__':
    unittest.main()