import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk memuat model dan data
@st.cache_resource
def load_models_and_data():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    with open('tfidf_matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)
    with open('cosine_similarity_matrix.pkl', 'rb') as file:
        cosine_sim = pickle.load(file)
    books_data = pd.read_csv('merged_books_data.csv')
    return tfidf_vectorizer, tfidf_matrix, cosine_sim, books_data

tfidf_vectorizer, tfidf_matrix, cosine_sim, books_data = load_models_and_data()

# Fungsi preprocessing teks
def preprocess_text(text):
    return text.lower()  # Implementasikan preprocessing tambahan jika diperlukan

# Fungsi untuk rekomendasi berdasarkan kata kunci
def get_recommendations_by_category(keyword, tfidf_vectorizer, tfidf_matrix, books_data, threshold=0.5):
    # Preprocess kata kunci (kategori)
    processed_keyword = preprocess_text(keyword)

    # Ubah kata kunci kategori menjadi vektor TF-IDF
    keyword_vector = tfidf_vectorizer.transform([processed_keyword])

    # Hitung cosine similarity antara kata kunci dan kategori semua buku
    sim_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()

    # Filter buku berdasarkan similarity >= threshold
    recommended_indices = [i for i, score in enumerate(sim_scores) if score >= threshold]

    # Ambil detail buku yang direkomendasikan
    recommended_books = books_data.iloc[recommended_indices].copy()
    recommended_books['similarity_score'] = [sim_scores[i] for i in recommended_indices]

    # Urutkan berdasarkan similarity score
    return recommended_books.sort_values(by='similarity_score', ascending=False)

# Aplikasi Streamlit untuk pencarian berdasarkan kategori
st.title("Sistem Rekomendasi Buku Berdasarkan Kata Kunci")

# Input kata kunci kategori
keyword = st.text_input("Masukkan kata kunci:")
threshold = st.slider("Threshold Similarity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

# Tombol untuk mencari rekomendasi
if st.button("Cari Rekomendasi"):
    if keyword:
        recommendations = get_recommendations_by_category(
            keyword, tfidf_vectorizer, tfidf_matrix, books_data, threshold
        )
        if not recommendations.empty:
            st.write("Hasil Rekomendasi:")
            st.dataframe(recommendations[['title', 'authors', 'description', 'categories', 'similarity_score']])
        else:
            st.warning("Tidak ada buku yang memenuhi kriteria.")
    else:
        st.error("Masukkan kategori terlebih dahulu!")
