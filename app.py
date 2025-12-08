from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import time

# Internal Modules
from database import init_db, get_db
from vector_engine import VectorEngine
from utils import generate_state_id, decode_state_id

app = Flask(__name__)
CORS(app)

# Initialize
init_db()
engine = VectorEngine()

# Load Client Matrix into memory (optional cache for non-send_file logic)
CLIENT_DATA_FILE = "client_matrix.json"
client_json_cache = None

if os.path.exists(CLIENT_DATA_FILE):
    with open(CLIENT_DATA_FILE, "r") as f:
        client_json_cache = json.load(f)

@app.route('/')
def serve_index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html'))

@app.route('/api/matrix', methods=['GET'])
def get_matrix():
    if os.path.exists(CLIENT_DATA_FILE):
        # We attach a specific Cache-Control header.
        # public: Indicates CDNs (Cloudflare) are allowed to store this.
        # max-age=3600: Cache for 1 hour. 
        # Cloudflare will check back with your server every hour to see if the file changed.
        response = send_file(CLIENT_DATA_FILE, mimetype='application/json')
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
        
    return jsonify({"error": "Matrix not generated"}), 500

@app.route('/api/vector', methods=['GET'])
def get_vector():
    word = request.args.get('word', '').lower().strip()
    vec = engine.get_vector(word)
    if vec is not None: return jsonify(vec.tolist())
    return jsonify(None), 404

@app.route('/api/suggestions', methods=['GET'])
def suggestions():
    word = request.args.get('word', '').lower().strip()
    neighbors, wildcards = engine.get_suggestions(word)
    return jsonify({"neighbors": neighbors, "wildcards": wildcards})

# --- RESTORED DATABASE ROUTES ---

@app.route('/api/save', methods=['POST'])
def save_relationship():
    data = request.json
    start, end = data.get('start_word'), data.get('end_word')
    depth = min(int(data.get('depth', 1)), 5)
    test_words = data.get('test_words', [])[:10]
    words_list = [r['word'] for r in test_words if engine.is_valid(r.get('word'))]
    
    if not engine.is_valid(start) or not engine.is_valid(end) or not words_list: 
        return jsonify({"error": "Invalid input"}), 400
    
    rel_id = generate_state_id(start, end, depth, words_list, engine.input_map)
    words_str = ",".join(words_list)
    
    conn = get_db()
    try:
        conn.execute("INSERT INTO relationships (id, start_word, end_word, depth, input_words, views, created_at) VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP) ON CONFLICT(id) DO UPDATE SET created_at = CURRENT_TIMESTAMP", (rel_id, start, end, depth, words_str))
        conn.commit()
        return jsonify({"id": rel_id, "status": "saved"})
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally: conn.close()

@app.route('/api/load', methods=['GET'])
def load_relationship():
    rel_id = request.args.get('id')
    conn = get_db()
    try:
        conn.execute("UPDATE relationships SET views = views + 1 WHERE id = ?", (rel_id,))
        conn.commit()
        rel = conn.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,)).fetchone()
        
        if rel:
            return jsonify({
                "start": rel['start_word'], 
                "end": rel['end_word'], 
                "depth": rel['depth'], 
                "words": rel['input_words'].split(',')
            })
            
        decoded = decode_state_id(rel_id, engine.id_to_word)
        if decoded:
             if engine.is_valid(decoded['start']) and engine.is_valid(decoded['end']):
                 return jsonify({
                    "start": decoded['start'],
                    "end": decoded['end'],
                    "depth": decoded['depth'],
                    "words": [w for w in decoded['path'] if engine.is_valid(w)]
                 })

        return jsonify({}), 404
    finally: conn.close()

@app.route('/api/library', methods=['GET'])
def get_library():
    search = request.args.get('search', '').strip()
    sort = request.args.get('sort', 'popular')
    conn = get_db()
    
    query = "SELECT id, start_word, end_word, depth, input_words, views, created_at FROM relationships"
    params = []
    
    if search:
        query += " WHERE (start_word LIKE ? OR end_word LIKE ?)"
        params.extend([f"{search}%", f"{search}%"])
    
    if sort == 'popular': query += " ORDER BY views DESC"
    else: query += " ORDER BY created_at DESC"
    
    query += " LIMIT 10"
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

if __name__ == '__main__':
    app.run(debug=False, port=5000)