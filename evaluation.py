from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load data,model
df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df.dropna(subset=['Plot', 'Title'])
model = SentenceTransformer("sbert-finetuned-movies")

# example evaluation dataset
eval_data = [
    ("Dreams inside dreams", "Inception"),
    ("A young lion fights to reclaim his kingdom", "The Lion King"),
    ("A magical school and a boy with a scar", "Harry Potter and the Sorcerer's Stone")
]

 #encode
corpus_embeddings = model.encode(df['Plot'].tolist(), convert_to_tensor=True)

def compute_top_k_accuracy(eval_data, corpus_embeddings, df, model, top_k=3):
    correct = 0
    for query_text, correct_title in eval_data:
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
        top_titles = [df.iloc[hit['corpus_id']]['Title'] for hit in hits]
        if correct_title in top_titles:
            correct += 1
    return correct / len(eval_data)

if __name__ == "__main__":
    top3_acc = compute_top_k_accuracy(eval_data, corpus_embeddings, df, model, top_k=3)
    print(f"Top-3 Accuracy: {top3_acc:.2%}")
