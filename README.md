# Semantic Search on Movie Plots

> A full-stack NLP project that implements semantic search using a fine-tuned Sentence-BERT model on Wikipedia movie plots. Powered by FAISS for efficient retrieval and deployed with a custom Streamlit UI.

---

##  Overview

This project allows users to input a natural language query (e.g., "dreams within dreams") and retrieve semantically relevant movies like *Inception*. 

It uses:
- **SBERT**: Fine-tuned SentenceTransformer for dense embeddings.
- **FAISS**: Fast Approximate Nearest Neighbors search.
- **Streamlit**: Custom UI with filters (Genre, Origin, Timeline).
- **Google T5-Base**: Used to generate query–document pairs.
- **Hugging Face Transformers**: For leveraging pretrained and custom fine-tuned models.

---

##  Key Concepts and Techniques

###  Semantic Search Architecture
- **Input Query ➔ SBERT Encoder ➔ FAISS Vector Search ➔ Top-k Movie Matches**
## complete arci-- https://ibb.co/1GPw0RKV
### 📄 Dataset
- **Source**: [`wiki_movie_plots_deduped.csv`](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- **Authors**: Loren E. Juliano, Remy Labesque (from the Kaggle repo)
- **Size**: ~81,000 English movie plots.

###  Query Generation and Fine-tuning
- Used a **Google T5-base-based Query Generator** to create ~40,000 pseudo query-document pairs.
- Trained SBERT on these generated queries using contrastive learning.
- This enhanced the model's retrieval performance on unseen natural language queries.

### Evaluation
- **Top-3 Accuracy** metric implemented.
- Sample Queries:
  - "Dreams inside dreams" → *Inception*
  - "A magical school and a boy with a scar" → *Harry Potter*

---

## 📚 Research Papers and References

 
###  Sentence-BERT (SBERT)
- **Paper**: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- **Authors**: Reimers & Gurevych
- **Key Idea**: Learn semantically meaningful sentence embeddings by fine-tuning BERT in a Siamese setup with contrastive loss.

### FAISS (Facebook AI Similarity Search)
- **Paper**: [FAISS: Efficient Similarity Search](https://faiss.ai/)
- **Authors**: Facebook AI Research
- **Key Idea**: Accelerated large-scale similarity search using quantization and GPU acceleration.

### Query Generator for SBERT Fine-Tuning
- **Model**: Google T5-base
- **Repo**: [sentence-transformers/query-gen](https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/query_generation/train_generator.py)
- **Usage**: Generated pseudo query-document pairs using T5-QueryGenerator → used to fine-tune SBERT.

---

##  Tech Stack

| Layer              | Tools/Frameworks                                         |
|-------------------|----------------------------------------------------------|
| Model             | SentenceTransformers, PyTorch, HuggingFace               |
| Vector Indexing   | FAISS (Flat Index, CPU)                                  |
| Frontend / UI     | Streamlit + Custom CSS Styling                           |
| Deployment        | Streamlit Cloud                                          |

---

##  Features
-  Natural language search across 80k+ movies.
- Filters: Genre, Origin/Ethnicity, Release Year Range.
-  Uses finetuned semantic embeddings + FAISS retrieval.
-  Stylish, custom Streamlit frontend with glowing inputs.

---

## How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/semantic_search_using_S-bert
cd semantic_search_using_S-bert
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## Project Structure
```
semantic_search_using_S-bert/
├── app.py                  # Streamlit UI code
├── evaluation.py           # Top-3 accuracy metrics
├── sbert-finetuned-movies/ # Finetuned model folder
│   └── model.safetensors
├── movie_plot.index        # FAISS index file
├── wiki_movie_plots_deduped.csv
├── generated_queries_all.tsv
├── requirements.txt
└── README.md
```

---

##  Live Demo
** [semanticsearchusing-s-bert.streamlit.app](https://semanticsearchusing-s-bert.streamlit.app)**

---

## 👤 Author
**Vishal Reddy Yadama**
- 👨‍💻 *Aspiring ML Engineer*
-  [LinkedIn](https://www.linkedin.com/in/vishal-reddy-45a00727b/) • [GitHub](https://github.com/rockzz123)


##  Future Work
- Add output re-ranking using cross-encoders or reranker transformers.
- Improve retrieval pipeline with ANN variants (HNSW, IVF).
-  Add support for multi-language plots or subtitles.
-  Integrate a relevance feedback loop using user preferences.

If you liked this project, consider giving it a ⭐ and feel free to fork or reach out with suggestions!

