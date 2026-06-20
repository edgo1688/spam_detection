# SMS Spam Classification Project

## Team Members
- Edwin Gómez
- Johan Sebastián Bonilla

## Project Overview
This project analyzes the **SMS Spam Collection** dataset and evaluates a pre-trained Hugging Face model for spam detection. It also includes a simple **Streamlit web application** that allows a user to classify a custom text message as **SPAM** or **HAM**.

## Objectives
1. Perform an **Exploratory Data Analysis (EDA)** of the SMS Spam Collection dataset.
2. Evaluate the pre-trained model [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier) on a random sample of 20 SMS messages.
3. Build a **Python web application** that uses the same model to classify user-entered text as SPAM or HAM.

## Repository Structure
```text
spam_detection/
├── README.md
├── requirements.txt
├── app/
│   └── app.py
└── notebook/
    └── sms_spam_eda_and_model_test.ipynb
```

## Dataset
- **Dataset:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains SMS messages labeled as:
- **ham**: legitimate / non-spam messages
- **spam**: unsolicited or promotional messages

## Model
- **Model:** `Goodmotion/spam-mail-classifier`
- **Hugging Face:** https://huggingface.co/Goodmotion/spam-mail-classifier

> Note: the selected model is a spam classifier hosted on Hugging Face. Since it was not specifically trained on the SMS Spam Collection dataset, the evaluation performed in this project should be considered **exploratory** rather than a benchmark of domain-specific performance.

## Project Workflow

### Part 1 — Notebook / Data Science Process
The notebook includes:
1. **Dataset loading and inspection**
2. **Data cleaning and preparation**
3. **Exploratory Data Analysis (EDA)**:
   - class distribution
   - message length distribution
   - word count distribution
   - character-based feature analysis
   - most frequent words in spam and ham
4. **Model testing**:
   - random selection of 20 messages
   - prediction using the Hugging Face model
   - confidence score per prediction
   - correctness check for each sample
5. **Evaluation summary**:
   - sample accuracy
   - spam precision / recall / F1 score

### Part 2 — Web Application
The Streamlit application:
- accepts a custom text message from the user
- runs the message through the same Hugging Face model
- displays the predicted label (**SPAM** or **HAM**)
- shows the confidence score
- highlights the result with a simple interface

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/edgo1688/spam_detection.git
cd spam_detection
```

### 2. Create and activate a virtual environment
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run

### Run the notebook
Open Jupyter Notebook or JupyterLab and run:
```bash
jupyter notebook
```
Then open:
```text
notebook/sms_spam_eda_and_model_test.ipynb
```

### Run the web app
```bash
streamlit run app/app.py
```

## Expected Output
- A notebook with the full EDA and model testing workflow.
- A web app where the user can paste a text message and receive a **SPAM/HAM** prediction.

## Notes and Limitations
- The Hugging Face model may need internet access the first time it is loaded so the weights can be downloaded.
- Since the model was not necessarily trained on SMS data, performance on SMS messages may differ from performance on email-like spam text.
- The “confidence” shown for each prediction corresponds to the score returned by the model for the predicted class.

## Suggested GitHub Description
**SMS spam classification project with EDA, Hugging Face model evaluation, and a Streamlit web app for SPAM/HAM prediction.**
