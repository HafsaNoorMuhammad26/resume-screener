import nltk

# Sab required NLTK data download karo
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Yahi missing resource hai
nltk.download('omw-1.4')   # Optional lekin helpful hai lemmatization ke liye

print("Sab NLTK data successfully download ho gaya!")