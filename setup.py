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
# Changed filenames to v2 to ensure we don't load old cache without the new words
INPUT_VOCAB_FILE = "vocab_v2.txt"
INPUT_VECTORS_FILE = "embeddings_v2.npy"
OUTPUT_VOCAB_FILE = "diverse_8000_vocab.txt"
OUTPUT_VECTORS_FILE = "diverse_8000_embeddings.npy"
SUGGESTIONS_FILE = "suggestions_index.npy"
CLIENT_DATA_FILE = "client_matrix.json" 

K_CLUSTERS = 8000
BATCH_SIZE = 1024 

# ==========================================
# 1. THE INVENTORY LISTS ("The Stuff")
# ==========================================

def get_dolch_nouns():
    """95 Nouns for Children (The Nursery)"""
    return {
        "apple", "baby", "back", "ball", "bear", "bed", "bell", "bird", "birthday", "boat",
        "box", "boy", "bread", "brother", "cake", "car", "cat", "chair", "chicken", "children",
        "coat", "corn", "cow", "day", "dog", "doll", "door", "duck", "egg", "eye",
        "farm", "farmer", "father", "feet", "fire", "fish", "floor", "flower", "game", "garden",
        "girl", "goodbye", "grass", "ground", "hand", "head", "hill", "home", "horse", "house",
        "kitty", "leg", "letter", "man", "men", "milk", "money", "morning", "mother", "name",
        "nest", "night", "paper", "party", "picture", "pig", "rabbit", "rain", "ring", "robin",
        "santa claus", "school", "seed", "sheep", "shoe", "sister", "snow", "song", "squirrel", "stick",
        "street", "sun", "table", "thing", "time", "top", "toy", "tree", "watch", "water",
        "way", "wind", "window", "wood"
    }

def get_swadesh_list():
    """207 Universal Concepts (The Primal/Tribal)"""
    return {
        "i", "you", "he", "we", "you", "they", "this", "that", "here", "there", "who", "what",
        "where", "when", "how", "not", "all", "many", "some", "few", "other", "one", "two",
        "three", "four", "five", "big", "long", "wide", "thick", "heavy", "small", "short",
        "narrow", "thin", "woman", "man", "human", "child", "wife", "husband", "mother", "father",
        "animal", "fish", "bird", "dog", "louse", "snake", "worm", "tree", "forest", "stick",
        "fruit", "seed", "leaf", "root", "bark", "flower", "grass", "rope", "skin", "meat",
        "blood", "bone", "fat", "egg", "horn", "tail", "feather", "hair", "head", "ear",
        "eye", "nose", "mouth", "tooth", "tongue", "fingernail", "foot", "leg", "knee", "hand",
        "wing", "belly", "guts", "neck", "back", "breast", "heart", "liver", "drink", "eat",
        "bite", "suck", "spit", "vomit", "blow", "breathe", "laugh", "see", "hear", "know",
        "think", "smell", "fear", "sleep", "live", "die", "kill", "fight", "hunt", "hit",
        "cut", "split", "stab", "scratch", "dig", "swim", "fly", "walk", "come", "lie",
        "sit", "stand", "turn", "fall", "give", "hold", "squeeze", "rub", "wash", "wipe",
        "pull", "push", "throw", "tie", "sew", "count", "say", "sing", "play", "float",
        "flow", "freeze", "swell", "sun", "moon", "star", "water", "rain", "river", "lake",
        "sea", "salt", "stone", "sand", "dust", "earth", "cloud", "fog", "sky", "wind",
        "snow", "ice", "smoke", "fire", "ash", "burn", "road", "mountain", "red", "green",
        "yellow", "white", "black", "night", "day", "year", "warm", "cold", "full", "new",
        "old", "good", "bad", "rotten", "dirty", "straight", "round", "sharp", "dull", "smooth",
        "wet", "dry", "correct", "near", "far", "right", "left", "at", "in", "with",
        "and", "if", "because", "name"
    }

def get_ogden_basic():
    """850 Words for International Trade (The Commercial Inventory)"""
    words = """
    account act addition adjustment advertisement agreement air amount amusement animal answer apparatus approval argument art attack attempt attention attraction authority back balance base behavior belief birth bit bite blood blow body brass bread breath brother building burn burst business butter canvas care cause chalk chance change cloth coal color comfort committee company comparison competition condition connection control cook copper copy cork cotton cough country cover crack credit crime crush cry current curve damage danger daughter day death debt decision degree design desire destruction detail development digestion direction discovery discussion disease disgust distance distribution division doubt drink driving dust earth edge education effect end error event example exchange existence expansion experience expert fact fall family father fear feeling fiction field fight fire flame flight flower fold food force form friend front fruit glass gold government grain grass grip group growth guide harbor harmony hate hearing heat help history hole hope hour humor ice idea impulse increase industry ink insect instrument insurance interest invention iron jelly join journey judge jump kick kiss knowledge land language laugh law lead learning leather letter level lift light limit linen liquid list look loss love machine man manager mark market mass meal measure meat meeting memory metal middle milk mind mine minute mist money month morning mother motion mountain move music name nation need news night noise note number observation offer oil operation opinion order organization ornament owner page pain paint paper part paste payment peace person place plant play pleasure point poison polish porter position powder power price print process produce profit property prose protest pull punishment purpose push quality question rain range rate ray reaction reading reason record regret relation religion representative request respect rest reward rhythm rice river road roll room rub rule run salt sand scale science sea seat secretary selection self sense servant sex shade shake shame shock side sign silk silver sister size skin skirt sky sleep slip slope smash smell smile smoke sneeze snow soap society son song sort sound soup space stage start statement steam steel step stitch stone stop story stretch structure substance sugar suggestion summer support surprise swim system talk taste tax teaching tendency test theory thing thought thunder time tin top touch trade transport trick trouble turn twist unit use value verse vessel view voice walk war wash waste water wave wax way weather week weight wind wine winter woman wood wool word work wound writing year
    angle ant apple arch arm army baby bag ball band basin basket bath bed bee bell berry bird blade board boat bone book boot bottle box boy brain brake branch brick bridge brush bucket bulb button cake camera card carriage cart cat chain cheese chest chin church circle clock cloud coat collar comb cord cow cup curtain cushion dog door drain drawer dress drop ear egg engine eye face farm feather finger fish flag floor fly foot fork fowl frame garden girl glove goat gun hair hammer hand hat head heart hook horn horse hospital house island jewel kettle key knee knife knot leaf leg library line lip lock map match monkey moon mouth muscle nail neck needle nerve net nose nut office orange oven parcel pen pencil picture pig pin pipe plane plate plough pocket pot potato prison pump rail rat receipt ring rod roof root sail school scissors screw seed sheep shelf ship shirt shoe skin skirt snake sock spade sponge spoon spring square stamp star station stem stick stocking stomach store street sun table tail thread throat thumb ticket toe tongue tooth town train tray tree trousers umbrella wall watch wheel whip whistle window wing wire worm
    able acid angry automatic awake bad beautiful bent bitter black blue boiling bright broken brown cheap chemical chief clean clear common complex conscious cut deep dependent early elastic electric equal fat female fertile first fixed flat free frequent full general good great grey hanging happy hard healthy high hollow ill important kind loose loud low male married material medical military mixed narrow natural necessary new normal old open opposite parallel past physical political poor possible present private probable public quick quiet ready red regular responsible right rough round sad safe same secret separate serious sharp short shut simple slow small smooth soft solid special sticky stiff straight strange strong sudden sweet tall thick tight tired true violent waiting warm wet white wide wise yellow young
    be come do get give go have keep let make may put say see seem send take waiting will yesterday
    about across after against among at before between by down for from in near of off on over through to under up with
    a all any every no other some such that the this
    again ago almost already also always anywhere as away backward even ever far forward here however just little much near not now often off often only or out perhaps quite rather seldom so sometimes still then there together very well where why yes
    """
    cleaned_words = set()
    for w in words.replace('\n', ' ').split():
        cleaned_words.add(w.strip().lower())
    return cleaned_words

# ==========================================
# 2. THE SOUL LIST ("The Archetypes")
# ==========================================

def get_civilization_archetypes():
    """
    The King/War/God List (Master Edition).
    ~1,300 High-salience concepts defining the human condition.
    Includes modern political structures, comprehensive dualities, and gender pairs.
    """
    return {
        # --- 1. HIERARCHY, ROLES & PEOPLE (Gender & Status Pairs) ---
        "king", "queen", "prince", "princess", "emperor", "empress", "monarch",
        "lord", "lady", "baron", "baroness", "duke", "duchess", "noble", "royal",
        "master", "mistress", "servant", "maid", "butler", "slave", "owner",
        "man", "woman", "boy", "girl", "gentleman", "lady", "sir", "madam",
        "father", "mother", "dad", "mom", "parent", "guardian",
        "son", "daughter", "brother", "sister", "sibling", "twin",
        "husband", "wife", "spouse", "partner", "widow", "widower",
        "grandfather", "grandmother", "grandson", "granddaughter", "ancestor", "heir",
        "uncle", "aunt", "nephew", "niece", "cousin", "kin",
        "hero", "heroine", "villain", "nemesis", "protagonist", "antagonist",
        "god", "goddess", "deity", "idol", "demigod",
        "priest", "priestess", "monk", "nun", "prophet", "oracle",
        "wizard", "witch", "sorcerer", "sorceress", "warlock", "healer",
        "actor", "actress", "host", "hostess", "patron", "artist", "muse",
        "hunter", "huntress", "gatherer", "farmer", "shepherd",
        "warrior", "soldier", "knight", "dame", "squire", "mercenary", "veteran",
        "captain", "commander", "admiral", "general", "lieutenant", "sergeant",
        "thief", "robber", "burglar", "pirate", "bandit", "outlaw", "criminal",
        "judge", "jury", "executioner", "lawyer", "attorney", "prosecutor", "defendant",
        "victim", "survivor", "witness", "suspect", "prisoner", "captive", "guard",
        "friend", "enemy", "ally", "foe", "rival", "stranger", "foreigner", "native",
        "citizen", "immigrant", "refugee", "exile", "tourist", "guest", "host",
        "child", "adult", "baby", "infant", "toddler", "teen", "teenager", "elder", "senior",
        "genius", "fool", "idiot", "savant", "expert", "novice", "amateur", "pro",
        "teacher", "student", "mentor", "pupil", "coach", "trainee", "apprentice",
        "doctor", "patient", "nurse", "surgeon", "therapist", "corpse", "skeleton",

        # --- 2. DUALITIES, STATES & QUALITIES ---
        # Time & Age
        "old", "young", "new", "ancient", "modern", "future", "past",
        "early", "late", "recent", "long", "short", "eternal", "temporary",
        "fast", "slow", "quick", "sudden", "gradual", "instant", "forever",
        # Physical Dimension
        "big", "small", "large", "tiny", "huge", "giant", "micro", "macro",
        "tall", "short", "wide", "narrow", "thick", "thin", "deep", "shallow",
        "heavy", "light", "full", "empty", "hollow", "solid",
        "high", "low", "top", "bottom", "peak", "base",
        # Condition
        "hot", "cold", "warm", "cool", "freezing", "burning", "boiling", "frozen",
        "wet", "dry", "damp", "soaked", "arid", "humid",
        "hard", "soft", "rough", "smooth", "sharp", "dull", "blunt",
        "clean", "dirty", "filthy", "pure", "polluted", "clear", "cloudy",
        "fresh", "rotten", "stale", "ripe", "raw", "cooked", "burnt",
        "strong", "weak", "powerful", "frail", "healthy", "sick", "ill",
        "alive", "dead", "living", "undead", "mortal", "immortal",
        "awake", "asleep", "conscious", "unconscious", "dreaming", "waking",
        "hungry", "full", "thirsty", "drunk", "sober", "addicted",
        "rich", "poor", "wealthy", "broke", "destitute", "prosperous",
        "cheap", "expensive", "free", "costly", "valuable", "worthless",
        "open", "closed", "shut", "locked", "unlocked", "broken", "fixed",
        # Abstract / Moral
        "good", "bad", "evil", "wicked", "holy", "cursed", "sacred", "profane",
        "right", "wrong", "true", "false", "real", "fake", "genuine", "artificial",
        "fair", "unfair", "just", "unjust", "legal", "illegal",
        "brave", "cowardly", "fearless", "afraid", "bold", "timid",
        "happy", "sad", "joyful", "miserable", "glad", "depressed",
        "angry", "calm", "furious", "peaceful", "violent", "gentle",
        "smart", "stupid", "wise", "ignorant", "clever", "clumsy",
        "beautiful", "ugly", "pretty", "hideous", "attractive", "repulsive",
        "kind", "cruel", "mean", "nice", "polite", "rude",
        "loud", "quiet", "silent", "noisy", "deafening", "muted",
        "visible", "invisible", "hidden", "exposed", "public", "private",
        "guilty", "innocent", "ashamed", "proud", "humble", "arrogant",
        "safe", "dangerous", "secure", "risky", "certain", "uncertain",

        # --- 3. WAR, POLITICS & STRUCTURE (Modern & Historical) ---
        # Roles
        "president", "vice president", "premier", "prime minister", "chancellor",
        "governor", "mayor", "senator", "congressman", "congresswoman", "representative",
        "councilor", "diplomat", "ambassador", "envoy", "delegate", "bureaucrat",
        "activist", "protester", "voter", "candidate", "incumbent", "lobbyist",
        "terrorist", "insurgent", "rebel", "revolutionary", "partisan", "guerilla",
        "dictator", "autocrat", "tyrant", "despot", "oligarch",
        # Institutions & Places
        "government", "administration", "regime", "cabinet", "parliament", "congress",
        "senate", "assembly", "council", "committee", "board", "ministry", "agency",
        "embassy", "consulate", "headquarters", "base", "outpost", "bunker",
        "nation", "country", "state", "province", "territory", "district", "county",
        "city", "village", "town", "capital", "metropolis", "municipality",
        "border", "frontier", "boundary", "zone", "sector", "checkpoint",
        # Actions & Concepts
        "election", "vote", "ballot", "campaign", "poll", "referendum",
        "law", "bill", "act", "policy", "regulation", "statute", "constitution",
        "veto", "impeachment", "scandal", "corruption", "bribe", "sanction", "embargo",
        "tax", "budget", "deficit", "inflation", "recession", "economy", "tariff",
        "war", "peace", "conflict", "crisis", "standoff", "ceasefire", "armistice",
        "treaty", "alliance", "coalition", "union", "federation", "confederation",
        "democracy", "republic", "monarchy", "theocracy", "communism", "capitalism",
        "socialism", "fascism", "liberalism", "conservatism", "anarchy",
        "rights", "freedom", "liberty", "justice", "equality", "sovereignty",
        "prison", "jail", "cell", "dungeon", "detention", "asylum",
        # Combat Equipment
        "weapon", "gun", "rifle", "pistol", "missile", "nuke", "bomb", "drone",
        "tank", "jet", "fighter", "bomber", "carrier", "submarine", "warship",
        "bullet", "ammo", "shell", "grenade", "mine", "explosive", "torpedo",
        "armor", "helmet", "vest", "shield", "radar", "satellite",

        # --- 4. COSMOS, ELEMENTS & NATURE ---
        "sun", "moon", "star", "planet", "comet", "asteroid", "meteor",
        "sky", "space", "universe", "galaxy", "vacuum", "void", "ether",
        "earth", "air", "fire", "water", "ice", "wind", "lightning", "thunder",
        "sea", "ocean", "river", "lake", "pond", "stream", "creek", "wave",
        "mountain", "hill", "valley", "canyon", "cliff", "cave", "tunnel",
        "forest", "jungle", "woods", "desert", "swamp", "marsh", "plain", "field",
        "rain", "snow", "hail", "storm", "hurricane", "tornado", "fog", "mist",
        "day", "night", "dawn", "dusk", "noon", "midnight", "twilight",
        "spring", "summer", "autumn", "fall", "winter", "season",
        "north", "south", "east", "west", "up", "down", "left", "right",
        "animal", "beast", "creature", "monster", "pet", "livestock",
        "lion", "tiger", "bear", "wolf", "fox", "dog", "cat", "horse",
        "bird", "eagle", "hawk", "crow", "dove", "owl", "snake", "dragon",
        "fish", "shark", "whale", "dolphin", "insect", "spider", "fly", "bee",
        "tree", "flower", "grass", "rose", "lily", "fruit", "vegetable", "seed",
        
        # --- 5. OBJECTS, TOOLS & TECHNOLOGY ---
        "machine", "engine", "motor", "robot", "computer", "phone", "screen",
        "car", "truck", "bus", "train", "plane", "ship", "boat", "bike",
        "wheel", "gear", "lever", "switch", "button", "wire", "cable",
        "key", "lock", "door", "window", "wall", "roof", "floor", "stairs",
        "table", "chair", "bed", "desk", "cabinet", "box", "bag", "chest",
        "pen", "paper", "book", "letter", "note", "map", "card", "ticket",
        "glass", "cup", "plate", "bowl", "spoon", "fork", "knife", "bottle",
        "food", "drink", "meat", "bread", "cheese", "wine", "beer", "medicine",
        "poison", "trash", "garbage", "waste", "treasure", "jewel", "gift",
        "art", "music", "song", "picture", "photo", "movie", "story",
        "circle", "square", "triangle", "line", "point", "shape", "form"
    }

def get_essential_vocabulary():
    """Combines all curated lists into one MUST-HAVE set."""
    print("--- Aggregating Essential Vocab ---")
    essential = set()
    essential.update(get_dolch_nouns())
    essential.update(get_swadesh_list())
    essential.update(get_ogden_basic())
    essential.update(get_civilization_archetypes())
    print(f"Total Forced Words: {len(essential)}")
    return essential

# ==========================================
# 2. MAIN LOGIC
# ==========================================

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
    essential_set = get_essential_vocabulary()

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

        # Merge internet words with our Essential list to make sure none are missed
        # (Though most likely the internet list has them, we want to be safe)
        internet_set = set([w.strip().lower() for w in all_words if w.strip().isalpha()])
        
        # We explicitly ensure our essential set is in the final list
        combined_set = internet_set.union(essential_set)
        
        clean_vocab = sorted(list(combined_set))
        print(f"Total Vocabulary Size: {len(clean_vocab)}")
        
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(clean_vocab, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        with open(INPUT_VOCAB_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_vocab))
        np.save(INPUT_VECTORS_FILE, embeddings)

    # PHASE 2: OUTPUT SELECTION (FORCED INCLUSION + CLUSTERING)
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
        print("\n--- 5. GENERATING CLUSTERS (HYBRID STRATEGY) ---")
        ensure_nltk_resources()
        
        # 1. Map Essential Words to Indices
        vocab_to_idx = {w: i for i, w in enumerate(clean_vocab)}
        
        forced_indices = []
        for w in essential_set:
            if w in vocab_to_idx:
                forced_indices.append(vocab_to_idx[w])
        
        print(f"Found {len(forced_indices)} essential words in embedding matrix.")
        
        # 2. Define the Candidate Pool for the remaining slots
        # We still want the "filler" words to be common words (not obscure nonsense)
        fdist = nltk.FreqDist(w.lower() for w in brown.words())
        most_common_words = set(list(fdist.keys())[:25000]) # Increased pool size slightly
        
        candidate_indices = []
        forced_set_indices = set(forced_indices)
        
        for idx, word in enumerate(clean_vocab):
            # Only add to candidate pool if it's NOT already forced and IS common
            if idx not in forced_set_indices and word in most_common_words:
                candidate_indices.append(idx)
                
        # 3. Calculate remaining slots
        remaining_slots = K_CLUSTERS - len(forced_indices)
        if remaining_slots < 0:
            remaining_slots = 0
            print("Warning: Essential list larger than K_CLUSTERS. Truncating not implemented.")

        print(f"Filling {remaining_slots} remaining slots using K-Means...")

        # 4. K-Means on the Candidates
        candidate_embeddings = embeddings[candidate_indices]
        kmeans = MiniBatchKMeans(n_clusters=min(remaining_slots, len(candidate_embeddings)), 
                                 random_state=42, 
                                 batch_size=256, 
                                 n_init='auto')
        kmeans.fit(candidate_embeddings)
        cluster_centers = kmeans.cluster_centers_

        # 5. Map Centers to real words
        filler_indices = []
        for center in tqdm(cluster_centers, desc="Mapping Centers"):
            # Compute distance to CANDIDATE embeddings only
            distances = 1 - np.dot(candidate_embeddings, center)
            closest_local_index = np.argmin(distances)
            global_index = candidate_indices[closest_local_index]
            filler_indices.append(global_index)

        # 6. Combine Lists
        final_indices = list(set(forced_indices + filler_indices)) # set just in case
        final_indices.sort()
        
        selected_words = [clean_vocab[i] for i in final_indices]
        selected_vectors = embeddings[final_indices]

        output_vectors = np.array(selected_vectors)
        output_vocab = selected_words

        print(f"Final Vocab Size: {len(output_vocab)}")

        with open(OUTPUT_VOCAB_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(selected_words))
        np.save(OUTPUT_VECTORS_FILE, output_vectors)

    # PHASE 3: SUGGESTIONS
    if not os.path.exists(SUGGESTIONS_FILE):
        print("\n--- 7. PRE-CALCULATING SUGGESTIONS ---")
        num_inputs = len(embeddings)
        suggestion_indices = np.zeros((num_inputs, 8), dtype=np.uint16)
        
        # Re-map output vectors for faster dot product
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
    if True:
        
        # Simple rounding to 4 decimal places to save ~30% space in JSON string representation
        rounded_vectors = np.round(output_vectors, 4)
        
        data = {
            "vocab": output_vocab,
            "vectors": rounded_vectors.flatten().tolist() 
        }
        with open(CLIENT_DATA_FILE, "w") as f:
            json.dump(data, f)
        print(f"✅ Created {CLIENT_DATA_FILE}")

    print("✅ Done!")

if __name__ == "__main__":
    run_setup()