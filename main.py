import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import streamlit as st

import json
import fitz  # PyMuPDF
from docx import Document

# Load the model
model = tf.keras.models.load_model('models/text_classification_model.h5')

# Load the tokenizer
with open('models/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Mapping Class Index to class names
class_dict = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def preprocess_text(text):
    # Preprocess the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    return padded

def predict_class(text):
    # Preprocess the input text
    padded = preprocess_text(text)
    
    # Predict the class
    prediction = model.predict(padded)
    class_index = np.argmax(prediction, axis=1)[0]
    class_name = class_dict[class_index]
    
    return class_name

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def read_doc(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text





st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ["Home", "Predict"])

if options == "Home":
    st.title("Document/Text Classification Model")
    st.write("""
    This application uses a bidirectional LSTM model to classify text into the following categories:
    - World
    - Sports
    - Business
    - Sci/Tech
    """)

elif options == "Predict":
    st.title("Document/Text Classification Prediction")
    
    file = st.file_uploader("Upload a file (PDF or DOC)", type=["pdf", "docx"])
    text_input = st.text_area("Or enter text to classify")
    
    if st.button("Predict"):
        if file is not None:
            if file.name.endswith(".pdf"):
                text = read_pdf(file)
            elif file.name.endswith(".docx"):
                text = read_doc(file)
            else:
                st.write("Unsupported file format")
                text = ""
        else:
            text = text_input
        
        if text:
            predicted_class = predict_class(text)
            st.info(f"The text is classified as: **{predicted_class}**")
        else:
            st.error("Please upload a file or enter text to classify.")
