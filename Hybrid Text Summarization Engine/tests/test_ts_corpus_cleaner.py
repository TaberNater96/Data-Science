import unittest
from CorpusCondenser.contractions import CONTRACTION_MAP
from CorpusCondenser.ts_corpus_cleaner import strip_html_tags, remove_accented_chars, expand_contractions, remove_special_characters, lowercase_text, preprocess_text

class TestTsCorpusCleaner(unittest.TestCase):

    def test_strip_html_tags(self):
        text = "<html><body><h1>Hello, world!</h1></body></html>"
        expected = "Hello, world!"
        self.assertEqual(strip_html_tags(text), expected)

    def test_remove_accented_chars(self):
        text = "résumé"
        expected = "resume"
        self.assertEqual(remove_accented_chars(text), expected)

    def test_expand_contractions(self):
        text = "I'm a test string."
        expected = "I am a test string."
        self.assertEqual(expand_contractions(text, CONTRACTION_MAP), expected)

    def test_remove_special_characters(self):
        text = "Hello, world! 123"
        expected = "Hello world 123"
        self.assertEqual(remove_special_characters(text), expected)

    def test_lowercase_text(self):
        text = "Hello, World!"
        expected = "hello, world!"
        self.assertEqual(lowercase_text(text), expected)

    def test_preprocess_text(self):
        text = "<html><body><h1>Hello, world!</h1></body></html>"
        expected = "hello world"
        self.assertEqual(preprocess_text(text), expected)

if __name__ == '__main__':
    unittest.main()