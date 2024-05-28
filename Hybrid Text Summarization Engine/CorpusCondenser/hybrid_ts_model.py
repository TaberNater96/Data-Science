import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from calendar import month_name
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Tuple, Any

class ExtractiveModel:
    """
    The model is designed for extractive text summarization, which involves selecting important sentences or 
    phrases from the original text and concatenating them to form a summary. It ranks sentences based on 
    various features like TF-IDF scores, cosine similarity of sentence embeddings, presence of named entities, 
    keywords, and topics.
    """
    def __init__(self, top_n: int) -> None:
        """
        Initialize the ExtractiveModel with a specified number of top sentences, adjustable.

        Parameters:
            top_n (int): Number of top sentences to extract for the summary.
        """
        self.top_n = top_n

    def rank_sentences(self, sentences: List[str], features: Dict[str, Any]) -> List[Tuple[float, str]]:
        """
        Rank sentences based on their combined scores of TF-IDF, cosine similarity, named entities, keywords, and topics.

        Parameters:
            sentences (List[str]): List of sentences to rank.
            features (Dict[str, Any]): Dictionary containing features like TF-IDF scores, sentence embeddings, named entities, keywords, and topics.

        Returns:
            List[Tuple[float, str]]: List of tuples containing scores and sentences, sorted by score in descending order.
        """
        tfidf_scores = features['tfidf']
        sentence_embeddings = features['embeddings']
        named_entities = features['entities']
        keywords = features['keywords']
        topics = features['topics']

        # Calculate cosine similarity matrix for sentence embeddings
        cosine_sim_matrix = cosine_similarity(sentence_embeddings)

        # Combine TF-IDF scores and cosine similarity of embeddings
        combined_scores = np.zeros(len(sentences))
        for i, sentence in enumerate(sentences):
            
            # Use sum of cosine similarities as a proxy for sentence importance
            combined_scores[i] += cosine_sim_matrix[i].sum()
            
            # Boost sentences containing named entities, keywords, and top topics
            if any(entity in sentence for entity, _ in named_entities):
                combined_scores[i] += 1.0  # boost for NER
            if any(keyword in sentence for keyword in keywords):
                combined_scores[i] += 0.5  # boost for keywords
            if any(topic in sentence for topic in topics['topics'].keys()):
                combined_scores[i] += 0.5  # boost for top topics

        # Sort the sentences based on their combined scores in descending order
        ranked_sentences = sorted(
            
            # Each sentence is paired with its score (as a tuple), and the list of tuples is sorted by score
            ((score, sent) for score, sent in zip(combined_scores, sentences)),
            key=lambda x: x[0], # sort by score in descending order
            reverse=True
        )

        return ranked_sentences

    def summarize(self, sentences: List[str], features: Dict[str, Any]) -> List[str]:
        """
        Generate a summary by extracting the top N ranked sentences.

        Parameters:
            sentences (List[str]): List of sentences to summarize.
            features (Dict[str, Any]): Dictionary containing features like TF-IDF scores, sentence embeddings, named entities, keywords, and topics.

        Returns:
            List[str]: List of top N ranked sentences.
        """
        # Rank the sentences based on their features
        ranked_sentences = self.rank_sentences(sentences, features)
        
        # Extract the top N sentences as the summary
        top_sentences = [sent[1] for sent in ranked_sentences[:self.top_n]]
        return top_sentences


class AbstractiveModel:
    """
    This model is designed for abstractive text summarization, which involves generating a new shorter text
    that conveys the most critical information from the original text. It uses transformer models from the 
    Hugging Face library for summarization.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', framework: str = 'pt') -> None:
        """
        Initialize the AbstractiveModel with a specified transformer model.

        Parameters:
            model_name (str): Name of the transformer model to use for summarization.
            framework (str): Framework to use (e.g., 'pt' for PyTorch or 'tf' for TensorFlow).
        """
        # Load the transformer model and tokenizer for summarization using Hugging Face pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = pipeline('summarization', model=model_name, framework=framework)

    def generate_summary(self, top_sentences: List[str]) -> str:
        """
        Generate a summary by concatenating and summarizing the top sentences.

        Parameters:
            top_sentences (List[str]): List of top sentences to summarize.

        Returns:
            str: Generated summary text.
        """
        text = " ".join(top_sentences)
        
        # Tokenize and summarize the text using the transformer model
        inputs = self.tokenizer(text, return_tensors='pt', max_length=1024, truncation=True) # uses a larger max_length to avoid truncation
        input_ids = inputs['input_ids']
        
        summaries = []
        max_chunk_size = 512 # maximum chunk size for summarization

        # Summarize in chunks to fit within the model's constraints
        for i in range(0, input_ids.shape[1], max_chunk_size):
            chunk = input_ids[:, i:i + max_chunk_size]
            summary = self.model(self.tokenizer.decode(chunk[0]), max_length=150, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        # Concatenate summaries to form the final summary
        final_summary = " ".join(summaries)
        return final_summary

class HybridSummarizationModel:
    """
    This model is designed to perform hybrid text summarization, which combines both extractive and abstractive 
    summarization models. The summarize method first uses the extractive model to select the most important 
    sentences from the input, and then feeds these sentences into the abstractive model to generate a final 
    summary.
    """
    def __init__(self, extractive_model: ExtractiveModel, abstractive_model: AbstractiveModel) -> None:
        """
        Initialize the HybridSummarizationModel with extractive and abstractive models.

        Parameters:
            extractive_model (ExtractiveModel): An instance of the ExtractiveModel class.
            abstractive_model (AbstractiveModel): An instance of the AbstractiveModel class.
        """
        # Initialize the extractive and abstractive summarization models
        self.extractive_model = extractive_model
        self.abstractive_model = abstractive_model

    def summarize(self, sentences: List[str], features: Dict[str, Any]) -> str:
        """
        Generate a hybrid summary by combining extractive and abstractive summarization methods.

        Parameters:
            sentences (List[str]): List of sentences to summarize.
            features (Dict[str, Any]): Dictionary containing features like TF-IDF scores, sentence embeddings, named entities, keywords, and topics.

        Returns:
            str: Generated hybrid summary text.
        """
        # Extract top sentences using the extractive model
        top_sentences = self.extractive_model.summarize(sentences, features)
        
        # Generate the final summary using the abstractive model
        final_summary = self.abstractive_model.generate_summary(top_sentences)
        
        # Improve the human interpretability of the summary
        human_readable_summary = self.make_human_interpretable(final_summary)
        return human_readable_summary

    def make_human_interpretable(self, summary: str) -> str:
        """
        Improve the human interpretability of the generated summary by fixing capitalization, replacing specific terms,
        and ensuring proper sentence formatting.

        Parameters:
            summary (str): The generated summary text.

        Returns:
            str: The human-interpretable summary text.
        """
        # Capitalize the first letter of each sentence
        summary = '. '.join(sentence.capitalize() for sentence in summary.split('. '))

        # Replace specific terms with more readable versions
        summary = summary.replace("cop26", "The 26th annual Conference of Parties")
        summary = re.sub(r'(\d+)c', r'\1°C', summary)
        summary = summary.replace("ndcs", "Nationally Determined Contributions")
        summary = summary.replace("co2", "CO2")
        summary = summary.replace("ghg", "greenhouse gas")

        # List of country names (can be expanded)
        countries = ["China", "Japan", "Korea", "United States", "India", "Brazil", "Russia", "Canada", "Australia", "Germany"]

        # Capitalize country names
        for country in countries:
            summary = re.sub(r'\b' + country.lower() + r'\b', country, summary)

        # Capitalize month names
        for month in month_name[1:]:  # month_name[1:] to skip the empty string at index 0
            summary = re.sub(r'\b' + month.lower() + r'\b', month, summary)

        return summary