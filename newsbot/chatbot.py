import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from transformers import pipeline
from langchain.memory import ConversationBufferMemory
import logging

logging.basicConfig(level=logging.INFO)

# Jina embedding model
class Embedded:
    def __init__(self):
        logging.info("Loading Jina embeddings model...")
        self.embed = pipeline(
            "feature-extraction",
            model="jinaai/jina-embeddings-v3",
            trust_remote_code=True
        )

    def generate_embeddings(self, text: str) -> list[float]:        # so this is basically changing like each small chunks embedding and then converting them into just one embedding so that the meaning is more similar
        token_embeddings = self.embed(text)[0]
        return np.mean(token_embeddings, axis=0).tolist()

# Falcon summarizer
class Summarizer:
    def __init__(self):
        logging.info("Loading summarization model...")
        self.pipe = pipeline("summarization", model="Falconsai/text_summarization")

    def summarize(self, text: str) -> str:                          # Summarizes long chunks into 100–150 token outputs.
        result = self.pipe(text, max_length=150, min_length=100, do_sample=False)
        return result[0]["summary_text"]

# PostgreSQL pgvector search --> same as in main.py
class ChunkDB:
    def __init__(self, dbname, user, password, host="localhost", port=5432):
        self.conn = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        self.cursor = self.conn.cursor()
        register_vector(self.conn)

    def search_similar_chunks(self, query_embedding, top_k=5):
        self.cursor.execute(
            """
            SELECT content, sourceurl
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """, (query_embedding, top_k)
        )
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()

# Main chatbot loop
if __name__ == "__main__":
    embedder = Embedded()                   # to convert question asked to vector
    summarizer = Summarizer()               # to give a summarized response
    db = ChunkDB(dbname="vecdb", user="user", password="123")           # get the database of embeddings made in main.py
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)      # to remember old conversation and maintain that in the next response

    print("Ask me something (Ctrl+C to quit):")
    try:
        while True:
            user_input = input("\nYou: ")

            # Save user message in memory
            memory.chat_memory.add_user_message(user_input)         # save the question asked in chat memory

            # Embed query and search
            embedding = embedder.generate_embeddings(user_input)        # creates the embedding of the user's question
            results = db.search_similar_chunks(embedding)               # checks similarity with out data base embeddings and forms result

            top_texts = []                                              # to make a list of the different article results
            source_urls = set()                                         #  to make a set of the urls used to make the summary
            for content, url in results:
                top_texts.append(content)
                source_urls.add(url)

            combined_text = " ".join(top_texts)                 # joining the closest srticle result into a combined text
            summary = summarizer.summarize(combined_text)       #giving a summarized combined text

            # Save bot response
            memory.chat_memory.add_ai_message(summary)          # save current chatbot response in memory

            print("\nChatbot Response:\n", summary)
            print("\nSources:")
            for url in source_urls:
                print(f"- {url}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        db.close()
