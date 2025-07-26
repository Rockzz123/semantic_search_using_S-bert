import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Set page 
st.set_page_config(page_title="Semantic Movie Search", layout="wide")

# Load data
df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df.dropna(subset=['Plot', 'Title', 'Release Year', 'Genre', 'Origin/Ethnicity'])
df['Genre'] = df['Genre'].fillna("Unknown")
df['Origin/Ethnicity'] = df['Origin/Ethnicity'].fillna("Unknown")
df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce').astype('Int64')
df = df.dropna(subset=['Release Year'])

# 10-year bins for Release Year
df['Year Bin'] = pd.cut(df['Release Year'], 
                        bins=list(range(int(df['Release Year'].min() // 10 * 10), 
                                       int(df['Release Year'].max() + 10), 10)))

# Load model and index
model = SentenceTransformer("sbert-finetuned-movies")
index = faiss.read_index("movie_plot.index")

# styles
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://c.tenor.com/9_NoAo1GeZsAAAAC/peaky-blinders.gif');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .block-container {
            background-color: rgba(0, 0, 0, 0.85);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }
        h1, h2, h3, .markdown-text-container {
            color: #E50914 !important;
        }
        div[data-baseweb="input"] {
            border: 2px solid red !important;
            border-radius: 10px;
            box-shadow: 0 0 10px red;
        }

        div[data-baseweb="input"] input {
            background-color: black;
            color: #ff4b4b;
            font-weight: bold;
            font-size: 18px;
            padding: 10px;
        }

        /* Glowing placeholder text */
        div[data-baseweb="input"] input::placeholder {
            color: #ff4b4b !important;
            font-style: italic;
            font-weight: bold;
            text-shadow: 0 0 5px #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

# Main content
title_html = """
    <div class='block-container'>
        <h1 style='text-align: center;'>🎬 Semantic Search on Movie Plots</h1>
        <p style='text-align: center; font-size: 1.2rem;'>Search your favourite movie with your query</p>
        <hr style='border: 1px solid #555;'>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Sidebar 
st.sidebar.markdown("## 📂 Filters")
bin_labels = [f"{int(b.left)} - {int(b.right)}" for b in df['Year Bin'].cat.categories]
year_filter = st.sidebar.selectbox("Release Year Range", ["All"] + bin_labels)
origin_filter = st.sidebar.selectbox("Origin/Ethnicity", ["All"] + sorted(df['Origin/Ethnicity'].unique()))
genre_filter = st.sidebar.selectbox("Genre", ["All"] + sorted(df['Genre'].unique()))

#  filters
df_filtered = df.copy()
if year_filter != "All":
    start, end = map(int, year_filter.split(" - "))
    df_filtered = df_filtered[(df_filtered['Release Year'] >= start) & (df_filtered['Release Year'] < end)]

if origin_filter != "All":
    df_filtered = df_filtered[df_filtered['Origin/Ethnicity'] == origin_filter]

if genre_filter != "All":
    df_filtered = df_filtered[df_filtered['Genre'].str.contains(genre_filter, case=False, na=False)]

#  query 
query = st.text_input(
    label="🔍 Enter a movie description or theme",
    placeholder="💡 Input your query to find your FAV MOVIE!",
    key="main_search_input"
)

# fetch movie info
def fetch_movie_info(dataframe_idx):
    info = df.iloc[dataframe_idx]
    return {
        'Title': info['Title'],
        'Year': info['Release Year'],
        'Genre': info['Genre'],
        'Origin': info['Origin/Ethnicity'],
        'Plot': info['Plot'][:500]
    }

#  search
def search(query_text, top_k=10):
    query_vector = model.encode([query_text], convert_to_numpy=True)
    scores, ids = index.search(query_vector, top_k)
    return [fetch_movie_info(idx) for idx in ids[0]]

# results
st.markdown("""<hr style='border: 1px solid #555;'>""", unsafe_allow_html=True)

if query:
    results = search(query)
    results = [r for r in results if r['Title'] in df_filtered['Title'].values]

    if results:
        st.subheader("🔎 Top Matching Movies")
        for res in results:
            with st.container():
                st.markdown(f"### 🎮 {res['Title']} ({res['Year']})")
                st.markdown(f"**Genre:** {res['Genre']} | **Origin:** {res['Origin']}")
                st.markdown(f"📝 {res['Plot']}")
                st.markdown("---")
    else:
        st.warning("No results matched the query and filters.")
else:
    st.subheader("📋 Movies Based on Filters")
    if not df_filtered.empty:
        for _, row in df_filtered.head(10).iterrows():
            with st.container():
                st.markdown(f"### 🎮 {row['Title']} ({row['Release Year']})")
                st.markdown(f"**Genre:** {row['Genre']} | **Origin:** {row['Origin/Ethnicity']}")
                st.markdown(f"📝 {row['Plot'][:500]}")
                st.markdown("---")
    else:
        st.warning("No movies found with the selected filters.")

st.markdown("""</div>""", unsafe_allow_html=True)
