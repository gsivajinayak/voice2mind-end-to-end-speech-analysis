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
|------------------|--------------|
| `main.ipynb` | Main notebook combining Speech-to-Text, Grammar Correction, Mental Health Detection, and TTS pipeline |
| `DistilBERT_model.ipynb` | Fine-tuned DistilBERT model for 7-class mental health detection |
| `biLSTM+pretrained_word2vec_model.ipynb` | BiLSTM using pretrained Word2Vec embeddings |
| `Classification Report for Mental Health Detection.pdf` | Performance report with metrics and evaluation plots |
| `Voice2Mind.pdf` | Project summary and research documentation |
| `requirements main.txt` | Python dependencies required to run the project |
| `README.md` | Main project documentation (this file) |
| `README main.md` | Alternate readme or notes version |

---

## 🧩 Pipeline Overview

| Stage | Model / Library | Function |
|-------|------------------|-----------|
| 🎧 **Speech-to-Text (STT)** |open-ai Whisper | Converts voice input into text |
| ✍️ **Grammar Correction** | T5 /mBART-50 | Refines and corrects sentence grammar |
| 🧠 **Mental Health Detection** | DistilBERT, BiLSTM, Ensemble ML | Classifies mental health category |
| 🔊 **Text-to-Speech (TTS)** | gTTS | Converts AI output into realistic speech |
| 🌐 **Interface** | Gradio | Interactive user interface for demo |

---

## 📊 Dataset

The dataset consists of **user-generated text posts** labeled into **7 mental health categories**:

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
- Balanced classes using oversampling and undersampling techniques

---

## 🔍 Exploratory Data Analysis (EDA)

We conducted:
- 📊 Class distribution analysis (bar & pie charts)
- ☁️ Word clouds for each mental health class
- 🧮 Text length distribution
- 🔡 Most frequent words per label


---

## 🧪 Models Implemented

### 1️⃣ **Speech-to-Text (Whisper)**
- Model: OpenAI Whisper (small/medium)
- Function: Converts audio → text with robust noise handling
- Accuracy: >95% on clean English speech

---

### 2️⃣ **Grammar Correction (Transformer)**
- Model: T5-small & mBART-50 fine-tuned for grammatical correction
- Example:
  - Input: *“He go to school yesterday.”*  
  - Output: *“He went to school yesterday.”*

---

### 3️⃣ **Mental Health Detection**
- Model: Fine-tuned **DistilBERT (Transformer)**
- Performance:
  - Accuracy: **91.06%**
  - Macro F1-score: **0.91**
- Additional models:
  - **BiLSTM + Word2Vec**
  - **Ensemble ML (XGBoost + RF + Logistic Regression)**

---

### 4️⃣ **Text-to-Speech (TTS)**
- Model: **Coqui TTS / pyttsx3**
- Function: Generates realistic human-like audio output  
- Supports **offline operation**

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

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- GPU (optional, for faster inference)
- Install dependencies:
  ```bash
  pip install -r requirements\ main.txt
