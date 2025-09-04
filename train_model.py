import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Load dataset - let's first check what columns it has
try:
    df = pd.read_csv('resume_dataset.csv')
    print("Dataset columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    print("First few rows:")
    print(df.head())
    
    # Identify the text column and category column
    # Common column names for resume datasets:
    text_column = None
    category_column = None
    
    # Check for common column names
    possible_text_columns = ['resume_text', 'text', 'Resume', 'resume', 'content', 'Resume_str']
    possible_category_columns = ['category', 'Category', 'class', 'Class', 'label', 'Label']
    
    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break
    
    for col in possible_category_columns:
        if col in df.columns:
            category_column = col
            break
    
    if not text_column or not category_column:
        # If we can't find the columns, let the user specify
        print(f"Available columns: {df.columns.tolist()}")
        text_column = input("Enter the name of the text column: ")
        category_column = input("Enter the name of the category column: ")
    
    print(f"Using text column: {text_column}")
    print(f"Using category column: {category_column}")
    
    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Check category distribution
    print("Category distribution:")
    print(df[category_column].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df[category_column], 
        test_size=0.2, 
        random_state=42,
        stratify=df[category_column]  # Maintain class distribution
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    with open('models/resume_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Also save the category mapping
    categories = df[category_column].unique().tolist()
    with open('models/categories.pkl', 'wb') as f:
        pickle.dump(categories, f)
    
    print("Model, vectorizer, and categories saved successfully!")
    
except FileNotFoundError:
    print("Resume dataset not found. Creating a mock model instead...")
    
    # Create a mock model for demonstration
    JOB_CATEGORIES = [
        "Data Scientist", 
        "Software Engineer", 
        "Web Developer", 
        "HR",
        "Business Development",
        "Health",
        "Advocate",
        "Arts"
    ]
    
    # Create a simple vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english'
    )
    
    # Create a simple logistic regression model
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000
    )
    
    # Create some dummy training data
    dummy_texts = []
    dummy_labels = []
    
    for category in JOB_CATEGORIES:
        for i in range(20):
            if category == "Data Scientist":
                text = f"python machine learning data science analytics {category} algorithm model"
            elif category == "Software Engineer":
                text = f"java python c++ software development engineering {category} code"
            elif category == "Web Developer":
                text = f"html css javascript web development frontend backend {category}"
            elif category == "HR":
                text = f"human resources recruitment talent acquisition {category} management"
            else:
                text = f"skills experience education {category} professional"
            
            dummy_texts.append(text)
            dummy_labels.append(category)
    
    # Preprocess and vectorize
    processed_texts = [preprocess_text(text) for text in dummy_texts]
    X = vectorizer.fit_transform(processed_texts)
    
    # Train the model
    model.fit(X, dummy_labels)
    
    # Save model and vectorizer
    with open('models/resume_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save categories
    with open('models/categories.pkl', 'wb') as f:
        pickle.dump(JOB_CATEGORIES, f)
    
    print("Mock model created and saved successfully!")