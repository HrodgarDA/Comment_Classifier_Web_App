
from Preprocessing import preprocess_text, predict_toxicity, load_resources, prepare_input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sopprime gli avvisi TensorFlow
import tensorflow as tf
import streamlit as st

# Load model and resources
@st.cache_resource
def load_model():
    model_path = '/Users/rugg/Documents/GitHub/Comment_Classifier_Web_App/LSMT_model.h5'
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_cached_resources():
    return load_resources()

model = load_model()
tokenizer, config = load_cached_resources()

# Streamlit app
st.title('Toxic Comment Classifier')

user_input = st.text_area("Enter a comment to classify:", "")

if st.button('Classify'):
    if user_input:
        input_sequence = prepare_input(user_input, tokenizer, config)
        prediction = model.predict(input_sequence)[0]
        result = {label: float(pred) for label, pred in zip(config['label_columns'], prediction)}
        
        st.write("Classification Results:")
        for label, prob in result.items():
            st.write(f"{label.capitalize()}: {prob:.2f}")
    else:
        st.write("Please enter a comment to classify.")