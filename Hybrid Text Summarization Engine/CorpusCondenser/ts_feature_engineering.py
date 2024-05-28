"""
By: Elijah Taber
Dated: May the 4th, 2024

This module, ts_feature_engineering.py, is designed to provide comprehensive feature engineering capabilities for a hybrid text 
summarization model using both extraction and abstraction feature engineering techniques, specifically tailored for summarizing 
complex, technical documents such as climate change reports. It includes a variety of feature extraction functions that prepare 
the text data by extracting meaningful and informative features, which are crucial for both extractive and abstractive 
summarization phases.

Features included are:
- TF-IDF scoring to highlight important terms.
- Sentence and word embeddings to capture semantic meanings.
- Named entity recognition for identifying and categorizing key information.
- Topic modeling to discern and quantify the main themes of the text.
- Keyword extraction to emphasize significant phrases and concepts.

The module concludes with a feature_engineering_pipeline function that facilitates the execution of all feature extraction methods, 
packaging the results for easy integration into summarization models.
"""

import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
from typing import Dict, List, Tuple

# Load NLP models
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')

# Initialize tools
tfidf_vectorizer = TfidfVectorizer()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
rake_nltk_var = Rake()

def extract_tfidf_features(sentences: List[str]) -> Dict[str, float]:
    """
    This function takes a list of sentences as input and returns a dictionary of words and their corresponding TF-IDF scores.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate how important
    a word is to a corpus in a collection or corpus. The importance increases proportionally to the number 
    of times a word appears in the corpus but is offset by the frequency of the word in the corpus.

    Parameters:
        sentences (List[str]): The list of sentences for which to calculate TF-IDF scores.

    Returns:
        Dict[str, float]: A dictionary where the keys are the words from the corpus and the values are their 
        corresponding TF-IDF scores.
    """
    # Join sentences to create a single document for TF-IDF calculation
    corpus = " ".join(sentences)
    tfidf_matrix = tfidf_vectorizer.fit_transform([corpus])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    return tfidf_scores

def extract_sentence_embeddings(sentences: List[str]) -> np.ndarray:
    """
    This function takes a list of sentences as input and returns sentence-level embeddings.
    
    Sentence embeddings are vector representations of sentences. These embeddings are generated using
    a pre-trained model (in this case, 'sentence_model') that has learned to map sentences to a 
    high-dimensional space where sentences with similar meanings are located close to each other.

    Parameters:
        sentences (List[str]): The list of sentences for which to generate embeddings.

    Returns:
        np.ndarray: A numpy array containing the sentence embeddings for the input sentences.
    """
    embeddings = sentence_model.encode(sentences)
    return embeddings

def extract_named_entities(corpus: str) -> List[Tuple[str, str]]:
    """
    This function takes a corpus as input and returns a list of named entities and their corresponding types.
    
    Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify 
    named entities in text into pre-defined categories such as person names, organizations, locations, medical
    codes, time expressions, quantities, monetary values, percentages, etc.

    Parameters:
        corpus (str): The corpus from which to extract named entities.

    Returns:
        List[Tuple[str, str]]: A list of tuples where the first element of each tuple is a named entity from 
        the corpus and the second element is the type of the named entity.
    """
    processed_corpus = nlp(corpus)
    entities = [(ent.text, ent.label_) for ent in processed_corpus.ents]
    return entities

def topic_modeling(corpus: str, n_topics: int = 5) -> Dict[str, any]:
    """
    This function performs topic modeling on a given corpus and returns the topic distribution for the corpus.
    
    Topic modeling is a type of statistical modeling for discovering the abstract "topics" that occur in a 
    collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to 
    classify text in a document to a particular topic. It builds a topic per document model and words per 
    topic model, modeled as Dirichlet distributions. This means it represents the document as a distribution
    of topics, instead of a distribution of words.

    Parameters:
        corpus (str): The corpus on which to perform topic modeling.
        n_topics (int, optional): The number of topics to be extracted from the corpus. Defaults to 5.

    Returns:
        Dict[str, any]: A dictionary containing topic distribution and top terms for each topic.
    """
    count_vectorizer = CountVectorizer(stop_words='english')
    corpus_term_matrix = count_vectorizer.fit_transform([corpus])
    
    lda = LDA(n_components=n_topics)
    lda.fit(corpus_term_matrix)
    
    topic_distribution  = lda.transform(corpus_term_matrix)[0]
    
    words = count_vectorizer.get_feature_names_out()
    topic_terms = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_10_terms = [words[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_terms[f"Topic {topic_idx + 1}"] = top_10_terms

    results = {
        'distribution': topic_distribution,
        'topics': topic_terms
    }
    
    return results

def extract_keywords(corpus: str) -> List[str]:
    """
    This function takes a corpus as input and returns a list of keywords and phrases.
    
    Keyword extraction is a process of extracting the most relevant words and expressions from text. 
    RAKE (Rapid Automatic Keyword Extraction) is a keyword extraction algorithm which sorts words by their 
    degree of importance.

    Parameters:
        corpus (str): The corpus from which to extract keywords.

    Returns:
        List[str]: A list of keywords and phrases extracted from the corpus.
    """
    rake_nltk_var.extract_keywords_from_text(corpus)
    return rake_nltk_var.get_ranked_phrases()

def feature_engineering_pipeline(sentences: List[str]) -> Dict[str, any]:
    """
    This function takes a list of sentences as input and returns a dictionary of various features extracted from the sentences.
    
    Feature engineering is the process of using domain knowledge to extract features from raw data. These 
    features can be used to improve the performance of machine learning algorithms. This function extracts 
    five types of features: TF-IDF scores, sentence embeddings, named entities, topics, and keywords.

    Parameters:
        sentences (List[str]): The list of sentences from which to extract features.

    Returns:
        Dict[str, any]: A dictionary where the keys are the names of the features and the values are the 
        extracted features.
    """
    corpus = " ".join(sentences)
    features = {}
    
    features['tfidf'] = extract_tfidf_features(sentences)
    features['embeddings'] = extract_sentence_embeddings(sentences)
    features['entities'] = extract_named_entities(corpus)
    features['topics'] = topic_modeling(corpus)
    features['keywords'] = extract_keywords(corpus)
    
    return features