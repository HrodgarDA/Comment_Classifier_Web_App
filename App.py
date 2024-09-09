import streamlit as st
import tensorflow as tf
from Preprocessing import preprocess_text, predict_toxicity, load_resources, prepare_input

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