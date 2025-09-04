# 📄 Resume Screening App  

## 🚀 Overview  

The **Resume Screening App** is a machine learning–powered tool designed to **automatically classify resumes** into job categories and provide a "fit score" for recruiters.  
It helps streamline the hiring process by reducing manual effort and ensuring consistent evaluation.  

Built using **Python, Scikit-Learn, and Streamlit**, this project demonstrates how Natural Language Processing (NLP) can be applied in real-world HR tech.  

---

## ✨ Features  

✅ Upload and process resumes (text-based format)  
✅ Classify resumes into predefined categories (e.g., Data Science, HR, IT, etc.)  
✅ Assign a confidence score for each prediction  
✅ User-friendly **Streamlit web interface**  
✅ Modular code for **retraining the model** with new datasets  
✅ Ready for **deployment** on Streamlit Cloud or Heroku  

---

## 📂 Project Structure  

```
resume-screener/
├── app.py                  # Streamlit frontend
├── train\_model.py          # Train and save ML model
├── utils/                  # Helper functions (preprocessing, text cleaning, etc.)
├── models/                 # Trained ML model + vectorizer
├── UpdatedResumeDataSet.csv # Training dataset
├── nltk\_download.py        # Downloads required NLTK resources
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment setup
├── setup.py                # Package setup
└── README.md               # Project documentation

```
---


## ⚙️ Installation


   ### 🔹 Option 1: Using Conda (Recommended)  

      ```bash
      git clone https://github.com/HafsaNoorMuhammad26/resume-screener.git
      cd resume-screener
      conda env create -f environment.yml
      conda activate resume-screener
      ```

   ### 🔹 Option 2: Using pip

   ```bash
   git clone https://github.com/HafsaNoorMuhammad26/resume-screener.git
   cd resume-screener
   pip install -r requirements.txt
   ```

   Also, download NLTK resources if required:


   ```bash
   python nltk_download.py
   ```
---

## 🖥️ Usage

### 1️⃣ Train the Model

```bash
python train_model.py
```

This script trains the model using `UpdatedResumeDataSet.csv` and saves the classifier in the `models/` folder.

### 2️⃣ Run the App

```bash
streamlit run app.py
```

The app will start locally. Open the provided URL in your browser (usually `http://localhost:8501`).

---

## 🌐 Deployment

### 🔹 Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub
3. Select this repo and deploy

---




## 🤝 Contributing

Contributions are welcome! 🚀

* Fork the repo
* Create a new branch (`feature-xyz`)
* Commit your changes
* Open a pull request

---


---

## 🙏 Acknowledgments

* **Hafsa Noor Muhammad** (Project Author)
Streamlit
 for the awesome framework

---

### ⭐ If you find this project useful, don’t forget to give it a star on GitHub! ⭐

