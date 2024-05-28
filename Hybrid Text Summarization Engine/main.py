import sys
import warnings
import spacy
import streamlit as st
import fitz # PyMuPDF
import os
import tempfile
from typing import List
from CorpusCondenser.pdf_processor import pdf_text_extractor
from CorpusCondenser.ts_corpus_cleaner import preprocess_text
from CorpusCondenser.ts_feature_engineering import feature_engineering_pipeline
from CorpusCondenser.hybrid_ts_model import ExtractiveModel, AbstractiveModel, HybridSummarizationModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load SpaCy NLP model for sentence segmentation
nlp = spacy.load('en_core_web_sm')

def process_pdf(pdf_path: str, start_page: int, end_page: int) -> str:
    """
    Processes a PDF file to extract and summarize the text using the full NLP pipeline.

    Parameters:
        pdf_path (str): Path to the PDF file.
        start_page (int): Starting page number for text extraction.
        end_page (int): Ending page number for text extraction.

    Returns:
        str: The generated summary of the PDF content.
    """
    # Step 1: Extract text from PDF
    raw_corpus = pdf_text_extractor(pdf_path, start_page, end_page)
    if not raw_corpus.strip():
        raise ValueError("Extracted text is empty. Please check the PDF content.")
    
    # Step 2: Preprocess the text
    cleaned_corpus = preprocess_text(raw_corpus)
    if not cleaned_corpus.strip():
        raise ValueError("Preprocessed text is empty. Please check the preprocessing steps.")

    # Step 3: Prepare sentences for summarization using SpaCy for sentence segmentation
    doc = nlp(cleaned_corpus)
    sentences = [sent.text for sent in doc.sents]
    if not sentences:
        raise ValueError("No sentences found after sentence segmentation. Please check the corpus content.")

    # Step 4: Perform feature engineering
    features = feature_engineering_pipeline(sentences)
    if not features or all(len(feature) == 0 for feature in features):
        raise ValueError("Feature engineering resulted in empty features. Please check the feature engineering pipeline.")

    # Step 5: Initialize models
    extractive_model = ExtractiveModel(top_n=5)
    abstractive_model = AbstractiveModel(model_name='facebook/bart-large-cnn', framework='pt')
    hybrid_model = HybridSummarizationModel(extractive_model, abstractive_model)

    # Step 6: Generate summary
    final_summary = hybrid_model.summarize(sentences, features)
    if not final_summary.strip():
        raise ValueError("Generated summary is empty. Please check the summarization model.")

    return final_summary

#############################################################################################################################################
#                                                     Streamlit Application                                                                 #
#############################################################################################################################################

# App layout
st.set_page_config(page_title="Hybrid Text Summarization Engine", page_icon=":memo:", layout="wide", initial_sidebar_state="expanded")

# Create a title and instructions for the app
st.markdown("<h1 style='text-align: center;'>Hybrid Text Summarization Engine Using NLP</h1>", unsafe_allow_html=True)
st.markdown("#### Upload your PDF file below to generate an AI summary! ⬇️")

# File uploader with PDF file specification
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process the uploaded PDF file and generate a summary
if uploaded_file is not None:
    
    # Save the uploaded PDF file to a temporary file to process
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.write("PDF uploaded successfully.")

    # Option for selecting start and end page numbers, not mandatory though
    st.markdown("<h4 style='font-weight: bold;'>Selecting a start and stop page makes the summary more accurate.</h4>", unsafe_allow_html=True)
    use_pages = st.checkbox("Select start and stop pages") # checkbox to enable page selection

    # If user wants to select pages, provide input fields for start and end pages, otherwise set to None
    if use_pages:
        start_page = st.number_input("Start Page", min_value=1, value=1)
        end_page = st.number_input("End Page", min_value=1, value=1)
    else:
        start_page = None
        end_page = None

    # Generate summary button
    if st.button("Generate Summary"):
        with st.spinner("Generating Summary..."):
            
            # Process the PDF file and generate a summary using the full NLP pipeline
            try:
                
                # If start and end pages are specified, process the PDF within that range
                if start_page and end_page:
                    summary = process_pdf(tmp_file_path, start_page, end_page)
                else:
                    # Default to processing the whole document if pages are not specified
                    summary = process_pdf(tmp_file_path, 1, float('inf')) # process pages from 1 to infinity
                    
                # Display that the summary has been generated successfully with a green outline
                st.success("Summary generated successfully!")
                
                # Display the generated summary
                st.write("## Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a PDF file to proceed.")
