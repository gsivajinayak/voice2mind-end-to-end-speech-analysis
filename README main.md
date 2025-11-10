# Voice2Mind: End-to-End Multilingual Speech Processing

Voice2Mind is an end-to-end multilingual NLP pipeline that converts speech into text, corrects grammar, translates between Hindi and English, detects mental health conditions, and converts the processed text back into speech.

## Features
- Speech-to-Text using OpenAI Whisper  
- Grammar correction using T5-base grammar model  
- Hindi–English translation using mBART-50  
- Mental health detection using fine-tuned DistilBERT  
- Text-to-Speech using Google gTTS  

## Workflow
Audio → Whisper STT → T5 Grammar Correction → mBART Translation → DistilBERT Sentiment Model → gTTS Speech Output

## Technologies Used
Python, Transformers, TensorFlow, Whisper, T5, mBART, DistilBERT, gTTS, Gradio

## Author
Developed by G. Sivaji 
B.Tech, CSE, IIT Bhubaneswar
