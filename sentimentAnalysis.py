import pandas as pd
import streamlit as st
import numpy as np
import torch

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModel, AutoTokenizer

from cleandata import Text_PreProcessing_util


def load_bert():

    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")

    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    return v_phobert, v_tokenizer

phobert, tokenizer = load_bert()

# Khởi tạo mô hình đã lưu
# Bạn cần thay đổi đường dẫn tới mô hình phù hợp
model_paths = {
    "Ensemble BiLSTM CNN Model": "model/ensemble_bilstm_cnn.keras",
    "LSTM Model": "model/lstm.keras",
    "BiLSTM Model": "model/bilstm.keras",
    "GRU Model": "model/gru.keras",
    "BiGRU Model": "model/bigru.keras",
    # "CNN Model": "model/cnn.keras",
    "Fusion BiLSTM CNN Model": "model/fusion_bilstm_cnn.keras",
    "Fusion BiGRU CNN Model": "model/fusion_bigru_cnn.keras",
    "Ensemble BiGRU CNN Model": "model/ensemble_bigru_cnn.keras",
}
models = {name: load_model(path) for name, path in model_paths.items()}

EMBEDDING_DIM = 768  # Dùng embedding vector của PhoBERT
index2class = {0: "Positive", 1: "Negative"}  # Mapping nhãn

def phobert_embed_sentence(padded, mask, model=phobert):
    padded = torch.tensor(padded).to(torch.long)
    mask = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids=padded, attention_mask=mask)[0]
    vector = last_hidden_states[:, 0, :].numpy()  # Lấy embedding vector
    return vector.flatten()

def predict_texts(texts, tokenizer, max_length=20, model=phobert):
    embedded_data = []
    texts = Text_PreProcessing_util(texts)


    for text in texts:
        tokenized_line = tokenizer.encode(text, max_length=max_length, truncation=True)
        padded_line = pad_sequences([tokenized_line], maxlen=max_length, padding='post', value=1)
        mask = np.where(padded_line == 1, 0, 1)
        embedded_line = phobert_embed_sentence(padded_line, mask)
        embedded_data.append(embedded_line)

    embedded_data = np.array(embedded_data)

    # Kiểm tra input shape của model
    input_shape = model.input_shape
    if len(input_shape) == 3:
        # Các mô hình như BiLSTM, BiGRU yêu cầu reshape 3D input
        input_data = embedded_data.reshape(embedded_data.shape[0], -1, EMBEDDING_DIM)
    elif len(input_shape) == 2:
        # Mô hình CNN hoặc các mô hình sử dụng input 2D
        input_data = embedded_data.reshape(embedded_data.shape[0], EMBEDDING_DIM)
    else:
        raise ValueError(f"Unsupported model input shape: {input_shape}")

    predictions = model.predict(input_data)
    predicted_classes = ((predictions > 0.5) + 0).ravel()  # Chuyển xác suất thành nhãn
    return predicted_classes

# Giao diện Streamlit
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
)


# Sidebar
with st.sidebar:
    st.header("Applications")
    app_mode = st.radio("Choose an application", ["Sentiment Analysis","Sentiment-Based Purchase Recommendation"])

if app_mode == "Sentiment Analysis":
    st.title("Vietnamese Review Sentiment Analysis 😊🤔😢")
    st.write("Phân tích cảm xúc của người mua hàng.")

    # Chọn mô hình
    selected_model_name = st.selectbox("Select a trained model", list(models.keys()))
    selected_model = models[selected_model_name]

    # Nhập văn bản đầu vào
    user_input = st.text_input("Input text", placeholder="Nhập văn bản tại đây...")
    if st.button("Analyze"):
        if user_input:
            try:
                # Dự đoán cảm xúc
                predicted_class = predict_texts([user_input], tokenizer, model=selected_model)
                # st.success(f"Sentiment Prediction: **{index2class[predicted_class[0]]}**")
                # Kiểm tra kết quả và hiển thị thông báo phù hợp
                if predicted_class[0] == 0:  # Nếu là Positive
                    st.success(f"Sentiment Prediction: **Positive**")
                else:  # Nếu là Negative
                    st.error(f"Sentiment Prediction: **Negative**")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter text to analyze.")
elif app_mode == "Sentiment-Based Purchase Recommendation":
    st.title("Sentiment-Based Purchase Recommendation")
    st.write("Khuyến nghị Mua hàng Dựa trên kết quả đánh giá Cảm xúc của Khách hàng")
    # Bạn có thể thêm chức năng Corpus Analysis ở đây
    # Chọn mô hình ngay từ đầu
    selected_model_name = st.selectbox("Select a trained model", list(models.keys()))
    selected_model = models[selected_model_name]

    # File uploader
    uploaded_file = st.file_uploader("Browse Corpus", type=["csv", "txt"], label_visibility="visible")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.write(f"File uploaded: {file_name}")

        sentences=[]
        # Read file into a pandas dataframe if it's a CSV
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.write("First few rows of the file:")
            st.write(df.head())  # Display first few rows of the uploaded CSV

            # Get list of sentences (assuming one column of text)
            if 'text' in df.columns:
                sentences = df['text'].tolist()
            else:
                st.error("No 'text' column found in CSV file.")

        # Read content if it's a TXT file
        elif file_name.endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8")
            sentences = content.split("\n")
            st.text(content)  # Display content of the uploaded text file
        else:
            st.error("Unsupported file type. Please upload CSV or TXT files.")

        if 'sentences' in locals() and sentences:
            st.write("Sentences in the file:")

            with st.spinner('Processing... Please wait while we analyze the sentences'):
                results = []

                positive_count = 0
                negative_count = 0

                for i, sentence in enumerate(sentences):
                    predicted_class = predict_texts([sentence], tokenizer, model=selected_model)
                    sentiment = "Positive" if predicted_class[0] == 0 else "Negative"
                    results.append({"Sentence": sentence, "Sentiment": sentiment})

                    if sentiment == "Positive":
                        positive_count += 1
                    else:
                        negative_count += 1

                # Chuyển danh sách kết quả thành DataFrame
                results_df = pd.DataFrame(results)

                # Tính tỷ lệ phần trăm
                total_count = len(sentences)
                positive_percentage = (positive_count / total_count) * 100
                negative_percentage = (negative_count / total_count) * 100

                # Hiển thị kết quả
                st.write("Results:")
                st.dataframe(results_df)
                st.success('Finished processing all sentences!')

                # Hiển thị tỷ lệ phần trăm cảm xúc tích cực và tiêu cực
                st.write(f"Positive Sentiment: {positive_percentage:.2f}%")
                st.write(f"Negative Sentiment: {negative_percentage:.2f}%")

                # Quyết định có nên mua hay không
                if positive_percentage > 70:
                    st.success("Recommendation: **Yes**, you should buy the product.")
                elif negative_percentage > 70:
                    st.error("Recommendation: **No**, you should not buy the product.")
                else:
                    st.info("Recommendation: **It's a mixed review**, consider other factors before deciding.")


        else:
            st.warning("No sentences found in the uploaded file.")
