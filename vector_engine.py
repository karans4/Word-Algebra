import numpy as np
import os

# --- CONFIGURATION ---
INPUT_VOCAB_FILE = "vocab_v2.txt"
INPUT_VECTORS_FILE = "embeddings_v2.npy"
OUTPUT_VOCAB_FILE = "diverse_5k_vocab.txt"
OUTPUT_VECTORS_FILE = "diverse_5k_embeddings.npy"
SUGGESTIONS_FILE = "suggestions_index.npy"

class VectorEngine:
    def __init__(self):
        self.input_map = {}
        self.input_matrix = None
        self.output_vocab = []
        self.output_map = {}
        self.output_matrix = None
        self.suggestions_index = None
        
        self.id_to_word = {} 
        self.load_resources()

    def load_resources(self):
        try:
            print("--- LOADING RESOURCES ---")
            if not os.path.exists(INPUT_VOCAB_FILE) or not os.path.exists(INPUT_VECTORS_FILE):
                print(f"CRITICAL: Missing Input files. Run setup.py")
                
            if os.path.exists(INPUT_VOCAB_FILE):
                with open(INPUT_VOCAB_FILE, "r", encoding="utf-8") as f:
                    self.input_map = {line.strip().lower(): i for i, line in enumerate(f)}
                    self.id_to_word = {v: k for k, v in self.input_map.items()}
            
            if os.path.exists(INPUT_VECTORS_FILE):
                self.input_matrix = np.load(INPUT_VECTORS_FILE) 
            
            if os.path.exists(OUTPUT_VOCAB_FILE):
                with open(OUTPUT_VOCAB_FILE, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f]
                    self.output_vocab = np.array(lines) 
                    self.output_map = {w.lower(): i for i, w in enumerate(lines)}

            if os.path.exists(OUTPUT_VECTORS_FILE):
                self.output_matrix = np.load(OUTPUT_VECTORS_FILE)
            
            # LOAD PRE-CALCULATED SUGGESTIONS
            if os.path.exists(SUGGESTIONS_FILE):
                self.suggestions_index = np.load(SUGGESTIONS_FILE)
                print(f"✅ Loaded Pre-calculated Suggestions: {self.suggestions_index.shape}")
            else:
                print("⚠️ Suggestions file not found. Run setup.py to pre-calculate for better performance.")

            print(f"✅ Ready. Input: {len(self.input_map)}, Output: {len(self.output_vocab)}")

        except Exception as e:
            print(f"Error loading resources: {e}")

    def get_vector(self, word):
        clean_w = word.lower().strip()
        if clean_w in self.input_map and self.input_matrix is not None:
            return self.input_matrix[self.input_map[clean_w]]
        return None

    def is_valid(self, word, max_len=30):
        if not word or len(word) > max_len: 
            return False
        return word.lower().strip() in self.input_map

    def get_suggestions(self, word):
        if not self.is_valid(word): return [], []
        
        clean_w = word.lower().strip()
        
        if self.suggestions_index is not None and clean_w in self.input_map:
            # SUPER FAST PATH: Pure Array Lookup
            idx = self.input_map[clean_w]
            
            # Retrieve row of 8 items [N, N, N, N, W, W, W, W]
            indices_row = self.suggestions_index[idx]
            
            # Split into Neighbors (First 4) and Wildcards (Last 4)
            neighbors = self.output_vocab[indices_row[:4]].tolist()
            wildcards = self.output_vocab[indices_row[4:]].tolist()
            
            return neighbors, wildcards
        
        # Fallback (Slow path if file missing or word valid but somehow not in map)
        neighbors = []
        vec = self.get_vector(word)
        if vec is not None and self.output_matrix is not None:
            scores = np.dot(self.output_matrix, vec)
            top_k = 4
            unsorted = np.argpartition(scores, -top_k)[-top_k:]
            idxs = unsorted[np.argsort(scores[unsorted])[::-1]]
            neighbors = self.output_vocab[idxs].tolist()

        wildcards = []
        if len(self.output_vocab) > 0:
            seed = abs(hash(clean_w)) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            wildcards = np.random.choice(self.output_vocab, 4, replace=False).tolist()
            
        return neighbors, wildcards