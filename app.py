import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import os

# NLTK data download - put in try block
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    st.warning("NLTK data download has issues. Using alternative methods.")

# Page configuration
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def simple_tokenize(text):
    """Simple tokenization without NLTK"""
    # Split words by spaces and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Try NLTK tokenization first
        try:
            tokens = word_tokenize(text)
        except:
            # If NLTK gives error, use simple tokenization
            tokens = simple_tokenize(text)
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            # Common English stopwords manually defined
            common_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in common_stopwords]
        
        # Lemmatization
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            # Simple stemming if lemmatization doesn't work
            tokens = [token[:-1] if token.endswith('s') else token for token in tokens]
        
        return ' '.join(tokens)
    
    except Exception as e:
        st.error(f"Text processing error: {e}")
        # Simple fallback - just lowercase and basic cleaning
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

@st.cache_resource
def load_model():
    try:
        with open('models/resume_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("Model file not found. Please run train_model.py first.")
        return None

@st.cache_resource
def load_vectorizer():
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except:
        st.error("Vectorizer file not found. Please run train_model.py first.")
        return None

@st.cache_resource
def load_categories():
    try:
        with open('models/categories.pkl', 'rb') as f:
            categories = pickle.load(f)
        return categories
    except:
        # Default categories if file not found
        return [
            "Data Scientist", 
            "Software Engineer", 
            "Web Developer", 
            "HR",
            "Business Development",
            "Health",
            "Advocate",
            "Arts"
        ]

def main():
    st.markdown('<h1 class="main-header">Resume Screener</h1>', unsafe_allow_html=True)
    
    # Load categories
    JOB_CATEGORIES = load_categories()
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        selected_category = st.selectbox("Select Target Job Role", JOB_CATEGORIES)
        st.info("Upload a resume to check its suitability for the selected role")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['pdf', 'docx'],
            help="Supported formats: PDF and DOCX"
        )
        
        if uploaded_file is not None:
            # Extract text
            if uploaded_file.type == "application/pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file type")
                return
            
            if raw_text and len(raw_text.strip()) > 0:
                # Preprocess text
                processed_text = preprocess_text(raw_text)
                
                # Display extracted text
                with st.expander("View Extracted Text"):
                    st.text_area("", raw_text, height=200)
                
                # Load model and vectorizer
                model = load_model()
                vectorizer = load_vectorizer()
                
                if model and vectorizer:
                    # Transform text
                    text_vector = vectorizer.transform([processed_text])
                    
                    # Predict category
                    prediction = model.predict(text_vector)
                    predicted_category = prediction[0]
                    
                    # Predict probability
                    probabilities = model.predict_proba(text_vector)[0]
                    max_probability = max(probabilities)
                    fit_score = round(max_probability * 100, 2)
                    
                    # Create probability dataframe
                    prob_df = pd.DataFrame({
                        'Category': JOB_CATEGORIES,
                        'Probability': probabilities
                    })
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Result box
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.write(f"**Predicted Role:** {predicted_category}")
                    st.write(f"**Fit Score:** {fit_score}%")
                    st.progress(fit_score/100)
                    
                    # Target role comparison
                    target_idx = JOB_CATEGORIES.index(selected_category)
                    target_score = round(probabilities[target_idx] * 100, 2)
                    
                    if predicted_category == selected_category:
                        st.success(f"This resume is a strong match for {selected_category}!")
                    else:
                        st.warning(f"This resume is predicted as {predicted_category}, not {selected_category}")
                        st.info(f"Fit score for {selected_category}: {target_score}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    with col2:
                        st.subheader("Detailed Analysis")
                        
                        # Probability chart
                        fig = px.bar(
                            prob_df, 
                            x='Probability', 
                            y='Category', 
                            orientation='h',
                            title='Probability by Job Category'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Word cloud
                        st.subheader("Resume Word Cloud")
                        wordcloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color='white'
                        ).generate(raw_text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                else:
                    st.error("Please run train_model.py first to create the model.")
            else:
                st.error("Could not extract text from the file. Please try another file.")
        else:
            # Show instructions when no file is uploaded
            with col2:
                st.info("""
                ### How to use:
                1. Select target job role from the sidebar
                2. Upload a resume (PDF or DOCX)
                3. View analysis results
                
                ### Before using:
                Run train_model.py first to create the model
                """)

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    main()