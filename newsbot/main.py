import newspaper
from newspaper import Article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import float32
from transformers import pipeline
import uuid
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dataclasses import dataclass

import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def generateEntries(src: str, limit: int = 1) -> list[str]:
    logging.info(f"Building newspaper source from: {src}")
    paper = newspaper.build(src, memoize_articles=False)
    urls = []
    logging.info(f"Found {len(paper.articles)} articles")
    for i, article in enumerate(paper.articles):
        if len(urls) >= limit:
            break
        try:
            article.download()
            article.parse()
            # Only add if article has content text (non-empty)
            if article.text and len(article.text) > 200:  # arbitrary minimum length
                urls.append(article.url)
                logging.debug(f"[{i}] Added article: {article.url}")
            else:
                logging.debug(f"[{i}] Skipped empty or short article: {article.url}")
        except Exception as e:
            logging.warning(f"[{i}] Failed to download or parse article: {e}")
            continue
    return urls

def articleRetrieve(src: str) -> str:
    logging.info(f"Retrieving article from URL: {src}")
    article = Article(src)
    try:
        article.download()
        article.parse()
        logging.debug("Article successfully downloaded and parsed")
        return article.text
    except Exception as e:
        logging.error(f"Failed to retrieve or parse article: {e}")
        return ""

def chunker(text: str) -> list[str]:
    logging.info("Splitting article into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs]
    logging.debug(f"Created {len(chunks)} chunks")
    return chunks

class Embedded:
    def __init__(self):
        logging.info("Initializing embedding pipeline...")
        self.embed = pipeline(
            "feature-extraction",
            model="jinaai/jina-embeddings-v3",
            trust_remote_code=True
        )
        logging.info("Embedding pipeline ready.")

    def generate_embeddings(self, text: str) -> list[float]:
        logging.debug("Generating embeddings for chunk...")
        token_embeddings = self.embed(text)[0]
        sentence_embedding = np.mean(token_embeddings, axis=0)
        logging.debug("Embedding generation complete.")
        return sentence_embedding.tolist()

@dataclass
class chunkentry:
    sourceurl: str
    content: str
    embedding: list[float]
    uid: str

class ChunkDB:
    def __init__(self, dbname, user, password, host="localhost", port=5432):
        logging.info("Connecting to PostgreSQL database...")
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        register_vector(self.conn)
        self.cursor = self.conn.cursor()
        logging.info("Connection successful. Creating table if not exists.")
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS chunks (
            uid UUID PRIMARY KEY,
            sourceurl TEXT,
            content TEXT,
            embedding VECTOR(1024)
        );
        """)
        self.conn.commit()

    def insert_chunk(self, entry: chunkentry):
        logging.debug(f"Inserting chunk UID: {entry.uid}")
        try:
            self.cursor.execute("""
            INSERT INTO chunks (uid, sourceurl, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uid) DO NOTHING;
            """, (entry.uid, entry.sourceurl, entry.content, entry.embedding))
            self.conn.commit()
            logging.debug("Chunk insertion successful.")
        except Exception as e:
            logging.error(f"Failed to insert chunk: {e}")

    def close(self):
        logging.info("Closing database connection.")
        self.cursor.close()
        self.conn.close()

# === Main Execution ===
if __name__ == "__main__":
    logging.info("Starting article processing pipeline...")
    embedder = Embedded()
    db = ChunkDB(dbname="vecdb", user="user", password="123")
    source = "http://cnn.com"
    urls = generateEntries(source, limit=1)

    for i, url in enumerate(urls):
        logging.info(f"[{i}] Processing article: {url}")
        doc = articleRetrieve(url)
        if not doc:
            logging.warning(f"Skipping empty article at {url}")
            continue
        chunks = chunker(doc)
        for j, chunk in enumerate(chunks):
            logging.debug(f"Processing chunk {j+1}/{len(chunks)}")
            embeddings = embedder.generate_embeddings(chunk)
            dbEntry = chunkentry(
                uid=str(uuid.uuid4()),
                sourceurl=url,
                content=chunk,
                embedding=embeddings
            )
            db.insert_chunk(dbEntry)
    db.close()
    logging.info("Pipeline complete.")

# # === Main Execution ===
# if __name__ == "__main__":
#     logging.info("Starting article processing pipeline...")
#     embedder = Embedded()
#     db = ChunkDB(dbname="vecdb", user="user", password="123")
#     url = "http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/"

#     doc = articleRetrieve(url)
#     chunks = chunker(doc)
#     for j, chunk in enumerate(chunks):
#         logging.debug(f"Processing chunk {j+1}/{len(chunks)}")
#         embeddings = embedder.generate_embeddings(chunk)
#         dbEntry = chunkentry(
#             uid=str(uuid.uuid4()),
#             sourceurl=url,
#             content=chunk,
#             embedding=embeddings
#         )
#         db.insert_chunk(dbEntry)
#     db.close()
#     logging.info("Pipeline complete.")
