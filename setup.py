import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import os
import sys
import nltk
from sklearn.cluster import MiniBatchKMeans
from nltk.corpus import brown
from tqdm import tqdm
import json

# --- CONFIGURATION ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
INPUT_VOCAB_FILE = "vocab.txt"
INPUT_VECTORS_FILE = "embeddings.npy"
OUTPUT_VOCAB_FILE = "diverse_5k_vocab.txt"
OUTPUT_VECTORS_FILE = "diverse_5k_embeddings.npy"
SUGGESTIONS_FILE = "suggestions_index.npy"
CLIENT_DATA_FILE = "client_matrix.json" # <-- NEW FOR FAT CLIENT

K_CLUSTERS = 5000
BATCH_SIZE = 1024 

def ensure_nltk_resources():
    print("--- Checking NLTK Resources ---")
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        print("Downloading NLTK 'brown' corpus...")
        nltk.download('brown')
    except Exception as e:
        print(f"Warning: NLTK check failed: {e}")

def run_setup():
    embeddings = None
    clean_vocab = None

    # PHASE 1: INPUTS
    if os.path.exists(INPUT_VOCAB_FILE) and os.path.exists(INPUT_VECTORS_FILE):
        print(f"✅ Found existing input files.")
    else:
        print("--- 1. DOWNLOADING & ENCODING ---")
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
        try:
            r = requests.get(url)
            all_words = r.text.splitlines()
        except Exception as e:
            print(f"Error downloading: {e}")
            return

        clean_vocab = sorted(list(set([w.strip().lower() for w in all_words if w.strip().isalpha()])))
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(clean_vocab, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        with open(INPUT_VOCAB_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_vocab))
        np.save(INPUT_VECTORS_FILE, embeddings)

    # PHASE 2: OUTPUTS
    if embeddings is None:
        print("Loading inputs...")
        embeddings = np.load(INPUT_VECTORS_FILE)
        with open(INPUT_VOCAB_FILE, "r", encoding="utf-8") as f:
            clean_vocab = f.read().splitlines()

    if os.path.exists(OUTPUT_VOCAB_FILE) and os.path.exists(OUTPUT_VECTORS_FILE):
         print(f"✅ Found existing output files.")
         output_vectors = np.load(OUTPUT_VECTORS_FILE)
         with open(OUTPUT_VOCAB_FILE, "r", encoding="utf-8") as f:
            output_vocab = f.read().splitlines()
    else:
        print("\n--- 5. GENERATING CLUSTERS ---")
        ensure_nltk_resources()
        fdist = nltk.FreqDist(w.lower() for w in brown.words())
        most_common_words = set(list(fdist.keys())[:20000])

        common_indices = [i for i, word in enumerate(clean_vocab) if word in most_common_words]
        common_embeddings = embeddings[common_indices]

        kmeans = MiniBatchKMeans(n_clusters=min(K_CLUSTERS, len(common_embeddings)), random_state=42, batch_size=256, n_init='auto')
        kmeans.fit(common_embeddings)
        cluster_centers = kmeans.cluster_centers_

        selected_words = []
        selected_vectors = []

        for center in tqdm(cluster_centers, desc="Mapping"):
            distances = 1 - np.dot(common_embeddings, center)
            closest_index_in_common = np.argmin(distances)
            original_index = common_indices[closest_index_in_common]
            selected_words.append(clean_vocab[original_index])
            selected_vectors.append(embeddings[original_index])

        output_vectors = np.array(selected_vectors)
        output_vocab = selected_words

        with open(OUTPUT_VOCAB_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(selected_words))
        np.save(OUTPUT_VECTORS_FILE, output_vectors)

    # PHASE 3: SUGGESTIONS
    if not os.path.exists(SUGGESTIONS_FILE):
        print("\n--- 7. PRE-CALCULATING SUGGESTIONS ---")
        num_inputs = len(embeddings)
        suggestion_indices = np.zeros((num_inputs, 8), dtype=np.uint16)
        
        for i in tqdm(range(0, num_inputs, BATCH_SIZE), desc="Batch Processing"):
            end_i = min(i + BATCH_SIZE, num_inputs)
            batch_vecs = embeddings[i:end_i]
            
            # Neighbors
            scores = np.dot(batch_vecs, output_vectors.T)
            top_k = np.argpartition(scores, -4, axis=1)[:, -4:]
            
            rows = np.arange(scores.shape[0])[:, None]
            sorted_order = np.argsort(scores[rows, top_k], axis=1)[:, ::-1]
            final_neighbors = top_k[rows, sorted_order]
            
            # Wildcards
            wildcards = np.random.randint(0, len(output_vectors), size=(end_i-i, 4))
            
            suggestion_indices[i:end_i] = np.hstack([final_neighbors, wildcards]).astype(np.uint16)

        np.save(SUGGESTIONS_FILE, suggestion_indices)

    # PHASE 4: CLIENT JSON EXPORT
    print("\n--- 8. EXPORTING CLIENT JSON ---")
    if not os.path.exists(CLIENT_DATA_FILE):
        # We perform precision reduction (float32 -> float16 or just rounding) to save space if needed
        # But for 5k words, standard float lists are fine (~15MB JSON).
        # To optimize, we flatten the list.
        
        data = {
            "vocab": output_vocab,
            # Flatten to 1D array for easier JS parsing
            "vectors": output_vectors.flatten().tolist() 
        }
        with open(CLIENT_DATA_FILE, "w") as f:
            json.dump(data, f)
        print(f"✅ Created {CLIENT_DATA_FILE}")

    print("✅ Done!")

if __name__ == "__main__":
    run_setup()