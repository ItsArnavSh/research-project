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

# i added a text summarizing model (Falconai) sinc earlier it was giving seperate output but with this we can get like a good compiled answer based on our prompt
class Summarizer:
    def __init__(self):
        logging.info("Loading summarization model...")
        self.pipe = pipeline("summarization", model="Falconsai/text_summarization")

    def summarize(self, text: str) -> str:
        logging.info("Generating summary of top chunks...")
        result = self.pipe(text, max_length=150, min_length=100, do_sample=False)
        return result[0]['summary_text']

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
    summarizer = Summarizer()
    db = ChunkDB(dbname="vecdb", user="user", password="123")

    print("Ask me something (Ctrl+C to quit):")
    try:
        while True:
            user_query = input("\nYou: ")
            embedding = embedder.generate_embeddings(user_query)
            results = db.search_similar_chunks(embedding)

            # Combine top chunks and summarize
            # Combine top chunks and summarize
            top_texts = []
            source_urls = set()

            for content, url in results:
                top_texts.append(content)
                source_urls.add(url)

            combined_text = " ".join(top_texts)
            summary = summarizer.summarize(combined_text)

            print("\nChatbot Response:\n", summary)
            print("\nSources:")
            for url in source_urls:
                print(f"- {url}")


    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        db.close()
