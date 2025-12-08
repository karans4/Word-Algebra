import sqlite3

DB_FILE = "saved_relations.db" 

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    # We only need to store the input configuration now.
    # The client will re-calculate the results instantly upon loading.
    conn.execute('''CREATE TABLE IF NOT EXISTS relationships (
        id TEXT PRIMARY KEY, 
        start_word TEXT, 
        end_word TEXT, 
        depth INTEGER,
        input_words TEXT, 
        views INTEGER DEFAULT 1, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('PRAGMA synchronous = NORMAL')
    conn.execute('PRAGMA journal_mode = WAL')
    conn.commit()
    conn.close()