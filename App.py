import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses TensorFlow warnings
import tensorflow as tf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Preprocessing import preprocess_text, predict_toxicity, load_resources, prepare_input

# ---------------------- Applying Custom Theme ----------------------

st.set_page_config(
    page_title="LSTM Classifier Web App",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- Model and Resource Loading ----------------------

@st.cache_resource
def load_model():
    """Load the LSTM model."""
    model_path = '/Users/rugg/Documents/GitHub/Comment_Classifier_Web_App/LSMT_model.h5'
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_cached_resources():
    """Load tokenizer and configuration."""
    return load_resources()

model = load_model()
tokenizer, config = load_cached_resources()

# ---------------------- Helper Functions ----------------------

def classify_single_comment(comment, threshold=0.55):
    """Classify a single comment and return results above threshold."""
    input_sequence = prepare_input(comment, tokenizer, config)
    prediction = model.predict(input_sequence)[0]
    result = {label: float(pred) for label, pred in zip(config['label_columns'], prediction) if float(pred) > threshold}
    return result

def display_classification_results(result):
    """Display classification results for classes above threshold."""
    if result:
        st.write("This comment has been classified as (probability [%]):")
        for label, prob in result.items():
            st.write(f"{label.capitalize()}: {prob:.2f}")
    else:
        st.write("The comment provided is ok")

def process_excel_file(df, threshold=0.55):
    """Process Excel file and return results dataframe with probabilities above threshold."""
    results = []
    for comment in df['comment']:
        result = classify_single_comment(comment, threshold)
        results.append(result)
    return pd.DataFrame(results)

# ---------------------- Visualization Functions ----------------------

def plot_class_distribution(results_df):
    """Plot the distribution of classes."""
    st.subheader("Distribution of classes")
    fig, ax = plt.subplots()
    results_df.mean().plot(kind='bar', ax=ax)
    plt.title("Average distribution of classes (prob > 0.55)")
    plt.ylabel("Average probability")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def display_most_toxic_comments(df, results_df):
    """Display the most toxic comments."""
    st.subheader("Most toxic comments")
    toxic_score = results_df.max(axis=1)
    most_toxic = df.loc[toxic_score.nlargest(5).index, 'comment']
    st.table(most_toxic)

def plot_comment_length_vs_toxicity(df, results_df):
    """Plot comment length vs toxicity."""
    st.subheader("Comment length vs toxicity")
    df['comment_length'] = df['comment'].str.len()
    toxic_score = results_df.max(axis=1)
    fig, ax = plt.subplots()
    plt.scatter(df['comment_length'], toxic_score)
    plt.xlabel("Comment length")
    plt.ylabel("Maximum toxicity score")
    st.pyplot(fig)

def generate_toxic_wordcloud(df, results_df):
    """Generate and display wordcloud of toxic comments."""
    st.subheader("Wordcloud of toxic comments")
    toxic_score = results_df.max(axis=1)
    toxic_comments = df.loc[toxic_score > 0.5, 'comment']
    if not toxic_comments.empty:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(toxic_comments))
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
    else:
        st.write("No toxic comments found for wordcloud generation.")

# ---------------------- Main App ----------------------

def main():
    st.title('Toxic Comment Classifier')

    input_option = st.radio("Choose input method:", ('Enter a comment', 'Upload Excel file'))

    if input_option == 'Enter a comment':
        user_input = st.text_area("Enter a comment to classify:", "")
        
        if st.button('Classify'):
            if user_input:
                result = classify_single_comment(user_input)
                display_classification_results(result)
            else:
                st.write("Please enter a comment to classify.")

    else:
        uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            if 'comment' in df.columns:
                if st.button('Classify all comments'):
                    results_df = process_excel_file(df)
                    st.write("This comment has been classified as:")
                    st.dataframe(results_df)

                    if not results_df.empty:
                        st.header("Insights")
                        plot_class_distribution(results_df)
                        display_most_toxic_comments(df, results_df)
                        plot_comment_length_vs_toxicity(df, results_df)
                        generate_toxic_wordcloud(df, results_df)
                    else:
                        st.write("No comments classified above the threshold.")
            else:
                st.error("The Excel file must contain a column named 'comment'.")

if __name__ == "__main__":
    main()