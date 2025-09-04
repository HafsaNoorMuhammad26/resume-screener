# Resume Screening App

An automated resume screening application that classifies resumes into job categories and provides a fit score.

## Setup

### Using Conda (Recommended)
1. `conda env create -f environment.yml`
2. `conda activate resume-screener`

### Using pip
1. `pip install -r requirements.txt`

## Running the Application

1. Train the model first (if you have the dataset):
   `python train_model.py`

2. Run the Streamlit app:
   `streamlit run app.py`

## Deployment

### Streamlit Cloud
1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account and select the repository
4. Deploy!

### Heroku
1. Install Heroku CLI
2. `heroku create your-app-name`
3. `git push heroku main`

## Project Structure
- `app.py` - Main Streamlit application
- `train_model.py` - Model training script
- `utils/` - Utility functions for text extraction and preprocessing
- `models/` - Saved model and vectorizer
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment specification