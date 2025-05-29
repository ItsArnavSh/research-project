import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from transformers import pipeline

import logging

logging.basicConfig(level=logging.INFO)

# same embedding model used in main.py
class Embedded:
    def __init__(self):
        logging.info("Loading Jina embeddings model...")
        self.embed = pipeline(
            "feature-extraction",
            model="jinaai/jina-embeddings-v3",
            trust_remote_code=True
        )

    def generate_embeddings(self, text: str) -> list[float]:
        token_embeddings = self.embed(text)[0]
        return np.mean(token_embeddings, axis=0).tolist()

# connected to pgvector-enabled PostgreSQL and register vector type
class ChunkDB:
    def __init__(self, dbname, user, password, host="localhost", port=5432):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.conn.cursor()
        register_vector(self.conn)

    # search top 5 similar chunks
    def search_similar_chunks(self, query_embedding, top_k=5):
        self.cursor.execute("""
            SELECT content, sourceurl
            FROM chunks
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (query_embedding, top_k))
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()

# Chatbot style output (just to view better we can improve this)
if __name__ == "__main__":
    embedder = Embedded()
    db = ChunkDB(dbname="vecdb", user="user", password="123")

    print("Ask me something (Ctrl+C to quit):")
    try:
        while True:
            user_query = input("\nYou: ")
            embedding = embedder.generate_embeddings(user_query)
            results = db.search_similar_chunks(embedding)

            print("\nBest Matching Content:")
            for i, (content, url) in enumerate(results):
                print(f"\n{i+1}. {content[:600]}...")  # trim long output
                print(f"Source: {url}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        db.close()
