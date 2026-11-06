import sqlite3

conn = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()
def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )
    """)

    conn.commit()

init_db()

def create_conversation():
    cursor.execute("INSERT INTO conversations (title) VALUES (?)", ("New Chat",))
    conn.commit()
    return cursor.lastrowid

def save_message(conversation_id, role, content):
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    conn.commit()

def get_qa_history(conversation_id):
    cursor.execute("""
        SELECT role, content 
        FROM messages 
        WHERE conversation_id=? 
        ORDER BY created_at
    """, (conversation_id,))

    rows = cursor.fetchall()

    history = []
    temp_question = None

    for role, content in rows:
        if role == "user":
            temp_question = content
        elif role == "assistant" and temp_question:
            history.append((temp_question, content))
            temp_question = None

    return history

def get_conversations():
    cursor.execute("""
        SELECT c.id, c.title
        FROM conversations c
        JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        ORDER BY c.created_at DESC
    """)
    return cursor.fetchall()

def update_conversation_title(conversation_id, title):
    cursor.execute(
        "UPDATE conversations SET title=? WHERE id=?",
        (title, conversation_id)
    )
    conn.commit()

def get_message_count(conversation_id):
    cursor.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id=?",
        (conversation_id,)
    )
    return cursor.fetchone()[0]

def delete_conversation(cid):
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    conn.commit()