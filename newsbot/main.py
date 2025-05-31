import newspaper
from newspaper import Article, article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg2.sql import NULL
from torch import float32
from transformers import pipeline
import uuid
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dataclasses import dataclass
import logging
import csv
from sentence_transformers import SentenceTransformer


# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def generateEntries(csv_file_path):
    links = []
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                links.append(row[0])
    return links


def articleRetrieve(src: str) -> Article:
    logging.info(f"Retrieving article from URL: {src}")
    article = Article(src)
    try:
        article.download()
        article.parse()
        logging.debug("Article successfully downloaded and parsed")
        return article
    except Exception as e:
        logging.error(f"Failed to retrieve or parse article: {e}")
        return Article("")

def chunker(article:Article) -> list[str]:
    logging.info("Splitting article into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([article.text])
    chunks = [doc.page_content for doc in docs]
    logging.debug(f"Created {len(chunks)} chunks")

    #chunks.append(article.title) Reason: Mostly Clickbait, and incomplete
    chunks.append(article.meta_description)

    return chunks

class Embedded:
    def __init__(self):
        logging.info("Initializing embedding pipeline...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def generate_embeddings(self, text: str) -> list[float]:
        logging.debug("Generating embeddings for chunk...")
        token_embeddings = self.model.encode(text)
        return token_embeddings.tolist()

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
        self.cursor = self.conn.cursor()
        logging.info("Connection successful. Creating table if not exists.")
        self._create_table()
        register_vector(self.conn)

    def _create_table(self):
        self.cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
        CREATE TABLE IF NOT EXISTS chunks (
            uid UUID PRIMARY KEY,
            sourceurl TEXT,
            content TEXT,
            embedding VECTOR(384),
            tsv tsvector
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
        """)
        self.conn.commit()


    def insert_chunk(self, entry: chunkentry):
        logging.debug(f"Inserting chunk UID: {entry.uid}")
        try:
            self.cursor.execute("""
            INSERT INTO chunks (uid, sourceurl, content, embedding, tsv)
            VALUES (%s, %s, %s, %s, to_tsvector('english', %s))
            ON CONFLICT (uid) DO NOTHING;
            """, (entry.uid, entry.sourceurl, entry.content, entry.embedding, entry.content))
            self.conn.commit()
            logging.debug("Chunk insertion successful.")
        except Exception as e:
            logging.error(f"Failed to insert chunk: {e}")
            self.conn.rollback()
    def hybrid_search(self, query_text, query_embedding, top_k=5) -> list[chunkentry]:
        self.cursor.execute("""
            SELECT uid, sourceurl, content, embedding,
                ts_rank(tsv, plainto_tsquery('english', %s)) AS bm25_score,
                (embedding <=> %s::vector) AS vec_distance
            FROM chunks
            ORDER BY bm25_score DESC, vec_distance ASC
            LIMIT %s;
        """, (query_text, query_embedding, top_k))

        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            uid, sourceurl, content, embedding_raw, bm25_score, vec_distance = row
            # Convert embedding to list if necessary
            if isinstance(embedding_raw, memoryview):
                embedding = list(embedding_raw)
            else:
                embedding = embedding_raw

            entry = chunkentry(
                uid=uid,
                sourceurl=sourceurl,
                content=content,
                embedding=embedding
            )
            results.append(entry)
        return results


    def close(self):
        logging.info("Closing database connection.")
        self.cursor.close()
        self.conn.close()

# # === Main Execution ===
if __name__ == "__main__":
#     # logging.info("Starting article processing pipeline...")
    embedder = Embedded()
    db = ChunkDB(dbname="vecdb", user="user", password="123")
    source = "news_urls.csv"
    urls = generateEntries(source)

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
    logging.info("Pipeline complete.")
    query_text = "Tariff with China"
    embedding = embedder.generate_embeddings(query_text)
    results = db.hybrid_search(query_text, embedding)
    for result in results:
        print(result.sourceurl)
    db.close()
