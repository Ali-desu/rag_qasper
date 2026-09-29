import re
import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32


class VectorStore:
    """Stores chunks, their embeddings and a BM25 index in one SQLite file.

    Three tables, all sharing the same integer id:
    - chunks     : text + metadata
    - vec_chunks : dense vectors (sqlite-vec)
    - fts_chunks : full-text index for BM25 (FTS5)

    In vec_chunks and fts_chunks, the built-in `rowid` is set to chunks.id,
    so a search result's rowid points straight back to its chunk.
    """

    def __init__(self, db_path="data/rag.db", dimension=384):
        """Open the database, load sqlite-vec, and create the tables if missing."""
        self.db_path = db_path
        self.dimension = dimension

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)

        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        self.create_tables()

    # ---------- setup ----------

    def create_tables(self):
        """Create the three tables if they don't exist yet (safe to call every time)."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY,
                    chunk_key   TEXT UNIQUE NOT NULL,
                    paper_id    TEXT NOT NULL,
                    section     TEXT,
                    chunk_index INTEGER NOT NULL,
                    text        TEXT NOT NULL
                )
            """)

            # paper_id is a metadata column so the search itself can filter by paper
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    embedding float[{self.dimension}],
                    paper_id text
                )
            """)

            # porter = English stemming ("datasets" matches "dataset")
            # paper_id is stored for filtering but not searched as text
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    text,
                    paper_id UNINDEXED,
                    tokenize = 'porter unicode61'
                )
            """)

    def reset(self):
        """Drop the three tables and create them again (makes ingestion re-runnable)."""
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS chunks")
            self.conn.execute("DROP TABLE IF EXISTS vec_chunks")
            self.conn.execute("DROP TABLE IF EXISTS fts_chunks")
        self.create_tables()

    # ---------- writing ----------

    def add_chunks(self, chunks, vectors):
        """Insert chunks into the three tables.

        chunks  : list of dicts from the chunker (id, paper_id, section, chunk_index, text)
        vectors : array of shape (len(chunks), dimension), same order as chunks

        Ids continue after the current highest id, so calling it several
        times (e.g. paper by paper) never reuses an id.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        start = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0] + 1

        rows_chunks, rows_vec, rows_fts = [], [], []
        for row_id, (chunk, vector) in enumerate(zip(chunks, vectors), start=start):
            rows_chunks.append((
                row_id, chunk["id"], chunk["paper_id"], chunk["section"],
                chunk["chunk_index"], chunk["text"],
            ))
            rows_vec.append((row_id, serialize_float32(vector.tolist()), chunk["paper_id"]))
            rows_fts.append((row_id, chunk["text"], chunk["paper_id"]))

        with self.conn:  # one transaction: all or nothing, and much faster
            self.conn.executemany(
                "INSERT INTO chunks (id, chunk_key, paper_id, section, chunk_index, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows_chunks,
            )
            self.conn.executemany(
                "INSERT INTO vec_chunks (rowid, embedding, paper_id) VALUES (?, ?, ?)",
                rows_vec,
            )
            self.conn.executemany(
                "INSERT INTO fts_chunks (rowid, text, paper_id) VALUES (?, ?, ?)",
                rows_fts,
            )

    # ---------- reading ----------

    def count(self):
        """Number of rows in each table. All three should be equal after ingestion."""
        return {
            table: self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("chunks", "vec_chunks", "fts_chunks")
        }

    def dense_search(self, query_vector, k=5, paper_id=None):
        """Vector search. Returns [(id, distance)], closest first.

        distance is L2. With normalized vectors it ranks exactly like cosine
        (cosine similarity = 1 - distance**2 / 2).
        """
        sql = "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        params = [serialize_float32(query_vector.tolist()), k]

        if paper_id is not None:
            sql += " AND paper_id = ?"
            params.append(paper_id)

        sql += " ORDER BY distance"
        return self.conn.execute(sql, params).fetchall()

    def keyword_search(self, query, k=5, paper_id=None):
        """BM25 search. Returns [(id, score)], best first.

        bm25() gives LOWER (more negative) scores for BETTER matches,
        so results are ordered ascending.
        """
        words = re.findall(r"\w+", query.lower())
        if not words:
            return []

        # quote each word so punctuation can't break the FTS5 syntax,
        # and join with OR: a plain space means AND, and a chunk rarely
        # contains every word of a question
        fts_query = " OR ".join(f'"{w}"' for w in words)

        sql = "SELECT rowid, bm25(fts_chunks) AS score FROM fts_chunks WHERE fts_chunks MATCH ?"
        params = [fts_query]

        if paper_id is not None:
            sql += " AND paper_id = ?"
            params.append(paper_id)

        sql += " ORDER BY score LIMIT ?"
        params.append(k)
        return self.conn.execute(sql, params).fetchall()

    def get_chunks(self, ids):
        """Full chunks (dicts) for a list of ids, in the SAME order as the ids."""
        ids = list(ids)
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, chunk_key, paper_id, section, chunk_index, text "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        # the database returns rows in any order: put them back in the order asked
        by_id = {
            row[0]: {
                "id": row[0],
                "chunk_key": row[1],
                "paper_id": row[2],
                "section": row[3],
                "chunk_index": row[4],
                "text": row[5],
            }
            for row in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    def close(self):
        """Close the connection."""
        self.conn.close()