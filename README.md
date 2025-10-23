# 🧠 Voice2Mind: End-to-End Speech_Analysis with Mental Health Detection

This repository presents an advanced **End-to-End Speech Analysis System** that integrates **Speech-to-Text (STT)**, **Grammar Correction**, **Mental Health Detection**, and **Text-to-Speech (TTS)** to build an intelligent and empathetic AI pipeline.

---

## 🎯 Project Objective

The goal of **Voice2Mind** is to enable a system that can:
- Listen to user audio,
- Transcribe it into text,
- Correct grammatical errors,
- Detect the user’s mental health state from their speech content,
- And finally, respond with a natural human-like voice.

This project bridges **speech understanding** and **mental state analysis**, making it useful for **healthcare, emotion analysis, and conversational AI**.

---

## 📂 Repository Contents

| File / Notebook | Description |
|-----------------|--------------|
| `Voice2Mind_Full_Pipeline.ipynb` | End-to-end notebook combining STT → Grammar → Mental Health → TTS |
| `stt_model/` | Whisper model files for Speech-to-Text |
| `grammar_model/` | Transformer-based Grammar Correction (T5/BERT) |
| `tts_model/` | Coqui TTS / pyttsx3 voice synthesis model |
| `DistilBERT_model.ipynb` | Fine-tuned DistilBERT model for 7-class mental health detection |
| `biLSTM+word2vec_model.ipynb` | BiLSTM using pretrained Word2Vec embeddings |
| `ensemble_model.ipynb` | Ensemble pipeline using Logistic Regression + RF + XGBoost |
| `balanced_dataset.csv` | Cleaned and balanced dataset for training |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation (this file) |

---

## 🧩 Pipeline Overview

| Stage | Model / Library | Function |
|-------|------------------|-----------|
| 🎧 **Speech-to-Text (STT)** | Whisper | Converts voice input into text |
| ✍️ **Grammar Correction** | T5 / BERT | Refines and corrects sentence grammar |
| 🧠 **Mental Health Detection** | DistilBERT, BiLSTM, Ensemble ML | Classifies mental health category |
| 🔊 **Text-to-Speech (TTS)** | Coqui TTS / pyttsx3 | Converts AI output into realistic speech |
| 🌐 **Interface** | Gradio | Interactive user interface for demo |

---

## 📊 Dataset

The dataset contains **user-generated text posts** labeled into 7 mental health categories:

| Label | Description |
|:------|:-------------|
| Anxiety | Fear, panic, overthinking |
| Bipolar | Mood swings, manic/depressive tone |
| Depression | Sadness, hopelessness, loss of interest |
| Normal | Neutral emotional tone |
| Personality Disorder | Inconsistent or extreme emotions |
| Stress | Overload, tension, pressure |
| Suicidal | Expressions of self-harm or suicide |

### 🧹 Preprocessing Steps:
- Removed stopwords, emojis, and URLs  
- Normalized casing and punctuation  
- Tokenized text sequences  
- Balanced classes (oversampling minority classes)

---

## 🔍 Exploratory Data Analysis (EDA)

We performed:
- Class distribution visualizations (bar + pie charts)  
- Word clouds for each mental health class  
- Text length histograms  
- Most frequent terms per label  

*(Add visuals under `/assets/` — like `class_distribution.png`, `wordcloud.png`, etc.)*

---

## 🧪 Models Implemented

### 1️⃣ Whisper (STT)
- Model: OpenAI Whisper (small/medium)  
- Function: Converts audio → text with robust noise handling  
- Accuracy: >95% transcription accuracy on clean speech  

### 2️⃣ Grammar Correction (Transformer)
- Model: T5-small / BERT fine-tuned for grammatical correction  
- Example:
  - Input: *“He go to school yesterday.”*  
  - Output: *“He went to school yesterday.”*

### 3️⃣ Mental Health Detection
- Model: **DistilBERT** (Transformer)  
- Accuracy: **91.06%**  
- Macro F1-score: **0.91**
- Additional models: BiLSTM + Word2Vec, Ensemble ML (XGBoost + RF)

### 4️⃣ Text-to-Speech (TTS)
- Model: Coqui TTS / pyttsx3  
- Output: High-quality speech synthesis (offline supported)  

---

## 📈 Mental Health Classification Results

| Class | Precision | Recall | F1-score |
|:--|:--:|:--:|:--:|
| Anxiety | 0.96 | 0.97 | 0.97 |
| Bipolar | 0.97 | 0.98 | 0.98 |
| Depression | 0.76 | 0.72 | 0.74 |
| Normal | 0.95 | 0.95 | 0.95 |
| Personality Disorder | 0.99 | 0.98 | 0.98 |
| Stress | 0.95 | 0.97 | 0.96 |
| Suicidal | 0.78 | 0.80 | 0.79 |

✅ **Overall Accuracy:** 91.06%  
✅ **Macro F1-score:** 0.91  

---

## ⚙️ How to Run

1️⃣ Clone this repository  
```bash
git clone https://github.com/<yourusername>/voice2mind-end-to-end-speech-analysis.git
cd voice2mind-end-to-end-speech-analysis
