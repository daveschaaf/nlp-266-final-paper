"""Shared, transactional window embeddings; experiments retain their own matrices."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import numpy as np


def encoder_identity(encoder, model):
    # Hash the actual loaded weights, rather than trusting a mutable model alias.
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    tokenizer = encoder.tokenizer.backend_tokenizer.to_str()
    modules = [dict(type=type(module).__qualname__, config=module.get_config_dict())
               for module in encoder.children() if hasattr(module, "get_config_dict")]
    return dict(model=model, weights_sha256=digest.hexdigest(),
                tokenizer_sha256=hashlib.sha256(tokenizer.encode()).hexdigest(),
                modules=modules, normalized=True, dtype="float32")


class WindowEmbeddingCache:
    def __init__(self, path, configuration, dimension):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.namespace = json.dumps(configuration, sort_keys=True, default=str)
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def key(self, text):
        return hashlib.sha256((self.namespace + "\0" + text).encode()).hexdigest()

    def get(self, text):
        row = self.connection.execute("SELECT vector FROM embeddings WHERE key=?", (self.key(text),)).fetchone()
        if row is None: return None
        vector = np.frombuffer(row[0], dtype=np.float32).copy()
        self.validate(vector)
        return vector

    def validate(self, vector):
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid shared embedding vector")

    def put(self, texts, vectors):
        if len(texts) != len(vectors): raise ValueError("Embedding count mismatch")
        rows = []
        for text, vector in zip(texts, vectors):
            vector = np.asarray(vector, dtype=np.float32)
            self.validate(vector)
            rows.append((self.key(text), vector.tobytes()))
        # A batch is either committed completely or rolled back on interruption.
        with self.connection:
            self.connection.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", rows)

    def close(self):
        self.connection.close()


def cached_encode(texts, encode, cache):
    started = time.perf_counter()
    records = [cache.get(text) for text in texts]
    missing = list(dict.fromkeys(text for text, record in zip(texts, records) if record is None))
    if missing:
        vectors = encode(missing)
        cache.put(missing, vectors)
    result = np.stack([record if record is not None else cache.get(text)
                       for text, record in zip(texts, records)])
    return result, dict(total=len(texts), hits=sum(r is not None for r in records),
                        encoded=len(missing), seconds=time.perf_counter()-started)
