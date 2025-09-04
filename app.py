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
from nltk.tokenize import word_tokenize
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

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
    .train-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .train-button:hover {
        background-color: #45a049;
    }
    .skill-pill {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 3px;
        display: inline-block;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Skills database for different job roles
SKILLS_DATABASE = {
    "Data Scientist": {
        "core": ["Python", "Machine Learning", "Data Analysis", "Statistics", "SQL", "Data Visualization"],
        "advanced": ["Deep Learning", "TensorFlow", "PyTorch", "Natural Language Processing", "Computer Vision", "Big Data"],
        "tools": ["Pandas", "NumPy", "Scikit-learn", "Jupyter", "Tableau", "Power BI", "Hadoop", "Spark"]
    },
    "Software Engineer": {
        "core": ["Java", "Python", "JavaScript", "C++", "Software Development", "Algorithms", "Data Structures"],
        "advanced": ["React", "Node.js", "Spring Boot", "Django", "Microservices", "API Development", "System Design"],
        "tools": ["Git", "Docker", "Kubernetes", "AWS", "Azure", "CI/CD", "Jenkins", "JIRA"]
    },
    "Web Developer": {
        "core": ["HTML", "CSS", "JavaScript", "React", "Web Development", "UI/UX", "Responsive Design"],
        "advanced": ["Node.js", "Express", "MongoDB", "REST API", "GraphQL", "TypeScript", "SASS", "LESS"],
        "tools": ["Git", "VS Code", "Chrome DevTools", "Webpack", "Bootstrap", "Figma", "Adobe XD"]
    },
    "HR": {
        "core": ["Recruitment", "Talent Acquisition", "Employee Relations", "HR Policies", "Interviewing", "Onboarding"],
        "advanced": ["Performance Management", "Compensation", "Benefits", "HRIS", "Compliance", "Talent Management", "Succession Planning"],
        "tools": ["LinkedIn Recruiter", "ATS", "Payroll Systems", "MS Office", "HR Analytics", "Workday", "SAP HR"]
    },
    "Business Development": {
        "core": ["Market Research", "Lead Generation", "Client Acquisition", "Sales", "Negotiation", "Relationship Management"],
        "advanced": ["Strategic Planning", "Partnership Development", "Market Analysis", "Competitive Intelligence", "CRM Management", "Sales Forecasting"],
        "tools": ["CRM Software", "Salesforce", "HubSpot", "LinkedIn Sales Navigator", "MS Office", "Google Analytics"]
    },
    "Health": {
        "core": ["Patient Care", "Medical Knowledge", "Clinical Skills", "Health Assessment", "Treatment Planning", "Medical Documentation"],
        "advanced": ["Specialized Procedures", "Diagnostic Skills", "Therapeutic Techniques", "Healthcare Management", "Medical Research", "Public Health"],
        "tools": ["Electronic Health Records", "Medical Software", "Diagnostic Equipment", "Telehealth Platforms", "Medical Databases", "Healthcare Apps"]
    },
    "Advocate": {
        "core": ["Legal Research", "Case Analysis", "Client Counseling", "Drafting", "Litigation", "Legal Documentation"],
        "advanced": ["Contract Law", "Corporate Law", "Intellectual Property", "Civil Law", "Criminal Law", "Alternative Dispute Resolution"],
        "tools": ["Legal Databases", "MS Office", "Document Management", "E-filing", "Case Management Software", "Legal Research Tools"]
    },
    "Arts": {
        "core": ["Creative Thinking", "Visual Communication", "Design Principles", "Color Theory", "Typography", "Sketching"],
        "advanced": ["Digital Illustration", "Photo Editing", "3D Modeling", "Animation", "Art Direction", "Brand Identity"],
        "tools": ["Adobe Photoshop", "Adobe Illustrator", "Adobe InDesign", "Procreate", "Blender", "CorelDRAW"]
    }
}

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

def extract_skills(text, job_role):
    """Extract skills from text based on job role"""
    text_lower = text.lower()
    matched_skills = []
    
    if job_role in SKILLS_DATABASE:
        # Check core skills
        for skill in SKILLS_DATABASE[job_role]["core"]:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                matched_skills.append(skill)
        
        # Check advanced skills
        for skill in SKILLS_DATABASE[job_role]["advanced"]:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                if skill not in matched_skills:
                    matched_skills.append(skill)
        
        # Check tools
        for skill in SKILLS_DATABASE[job_role]["tools"]:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                if skill not in matched_skills:
                    matched_skills.append(skill)
    
    return matched_skills

@st.cache_resource
def load_model():
    try:
        with open('models/resume_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("Model file not found. Please train the model first.")
        return None

@st.cache_resource
def load_vectorizer():
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except:
        st.error("Vectorizer file not found. Please train the model first.")
        return None

@st.cache_resource
def load_categories():
    try:
        with open('models/categories.pkl', 'rb') as f:
            categories = pickle.load(f)
        return categories
    except:
        # Default categories if file not found
        return list(SKILLS_DATABASE.keys())

def train_mock_model():
    """Create and train a mock model with sample data"""
    # Create sample job categories
    categories = list(SKILLS_DATABASE.keys())
    
    # Create sample training data
    sample_texts = []
    sample_labels = []
    
    for category in categories:
        # Generate sample resumes for each category
        for i in range(25):  # 25 samples per category
            # Get skills for this category
            core_skills = SKILLS_DATABASE[category]["core"]
            advanced_skills = SKILLS_DATABASE[category]["advanced"]
            tools = SKILLS_DATABASE[category]["tools"]
            
            # Select some skills for this sample
            num_core = min(3, len(core_skills))
            num_advanced = min(2, len(advanced_skills))
            num_tools = min(2, len(tools))
            
            selected_core = np.random.choice(core_skills, num_core, replace=False)
            selected_advanced = np.random.choice(advanced_skills, num_advanced, replace=False)
            selected_tools = np.random.choice(tools, num_tools, replace=False)
            
            # Create the text
            skills_text = ' '.join(list(selected_core) + list(selected_advanced) + list(selected_tools))
            text = f"{skills_text} {category} professional with experience"
            
            sample_texts.append(text)
            sample_labels.append(category)
    
    # Preprocess the texts
    processed_texts = [preprocess_text(text) for text in sample_texts]
    
    # Create and train the vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(processed_texts)
    
    # Create and train the model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X, sample_labels)
    
    # Save the model and vectorizer
    os.makedirs('models', exist_ok=True)
    
    with open('models/resume_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('models/categories.pkl', 'wb') as f:
        pickle.dump(categories, f)
    
    return model, vectorizer, categories

def main():
    st.markdown('<h1 class="main-header">Resume Screener</h1>', unsafe_allow_html=True)
    
    # Training section in sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Model training option
        st.subheader("Model Training")
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model... This may take a few moments."):
                model, vectorizer, categories = train_mock_model()
                st.success("Model trained successfully!")
        
        st.info("You can train the model or use the pre-trained one")
        
        # Load categories after potential training
        JOB_CATEGORIES = load_categories()
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
                    
                    # Extract skills for the predicted role
                    matched_skills = extract_skills(raw_text, predicted_category)
                    
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
                    st.write(f"**Extracted Job Role:** {predicted_category}")
                    st.write(f"**Fit Score:** {fit_score}%")
                    st.progress(fit_score/100)
                    
                    # Display matched skills
                    if matched_skills:
                        skills_html = "".join([f'<span class="skill-pill">{skill}</span>' for skill in matched_skills])
                        st.markdown(f"**Key Matched Skills:** {skills_html}", unsafe_allow_html=True)
                    else:
                        st.write("**Key Matched Skills:** No specific skills detected")
                    
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
                    st.error("Please train the model first using the button in the sidebar.")
            else:
                st.error("Could not extract text from the file. Please try another file.")
        else:
            # Show instructions when no file is uploaded
            with col2:
                st.info("""
                ### How to use:
                1. First, train the model using the button in the sidebar
                2. Select target job role from the sidebar
                3. Upload a resume (PDF or DOCX)
                4. View analysis results
                
                ### Note:
                The model is trained on sample data. For better accuracy, 
                use a real resume dataset.
                """)

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    main()