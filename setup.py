from setuptools import setup, find_packages

setup(
    name="resume-screener",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.22.0",
        "pandas==1.5.3",
        "numpy==1.24.3",
        "scikit-learn==1.2.2",
        "nltk==3.8.1",
        "python-docx==0.8.11",
        "PyPDF2==3.0.1",
        "plotly==5.14.1",
        "wordcloud==1.9.2",
    ],
)