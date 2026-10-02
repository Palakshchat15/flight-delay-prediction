"""Shared Postgres connection helper for the ML and dashboard scripts.

The read helpers keep the flight-level reads inside a 4 GB Docker VM: psycopg2 creates one
Python object per value, and holding ~2.2M rows that way (or keeping objects from every chunk
alive among freed ones, which fragments Python's allocator) costs gigabytes.
Read the unique string column (copy_text_column) BEFORE streaming the other columns: its
strings then fill fresh memory instead of the gaps the chunks' temporary objects leave behind.
"""
import ctypes
import gc
import io
import os

import numpy as np
import pandas as pd
import sqlalchemy


def get_engine():
    user = os.environ.get("POSTGRES_USER", "flight_user")
    password = os.environ.get("POSTGRES_PASSWORD", "flight_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5436")
    db = os.environ.get("POSTGRES_DB", "flight_delay")
    return sqlalchemy.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")


def read_sql_chunks(engine, sql, chunksize=200_000):
    """Yields the result in DataFrame chunks through a server-side cursor, so the whole result
    is never held as Python row objects at once."""
    with engine.connect().execution_options(stream_results=True) as conn:
        yield from pd.read_sql(sqlalchemy.text(sql), conn, chunksize=chunksize)


def concat_columns(chunks):
    """The columns of pd.concat(chunks, ignore_index=True), as {name: array}, for a generator of
    DataFrame chunks, without ever holding all chunks AND the result: each chunk is split into
    per-column arrays at once, and each result column is concatenated (pandas' own dtype rules,
    e.g. int64 + float32 chunks -> float64) and its pieces freed before the next. Build the frame
    with pd.DataFrame(columns, copy=False): columns stay separate blocks, nothing is copied."""
    pieces = {}
    for chunk in chunks:
        for c in chunk.columns:
            pieces.setdefault(c, []).append(chunk[c].to_numpy().copy())  # own memory, not the block
        del chunk
    return {c: pd.concat([pd.Series(a, copy=False) for a in pieces.pop(c)], ignore_index=True).to_numpy()
            for c in list(pieces)}


def shared_objects(s, canon):
    """Same values, but one Python object per distinct value across all chunks (`canon` is a
    dict shared by the chunks of one column). NULLs stay None."""
    codes, uniques = pd.factorize(s)
    if len(uniques) == 0:
        return s
    u = np.array([canon.setdefault(v, v) for v in uniques], dtype=object)
    return pd.Series(np.where(codes >= 0, u[codes], None), index=s.index, dtype=object)


def release_memory():
    """Hand freed memory back to the OS: glibc keeps freed heap pages (e.g. libpq's buffer of
    each fetched chunk) mapped until malloc_trim is called. No-op where glibc is not available."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def copy_text_column(engine, sql):
    """One text column of `sql` via COPY, as an object array of densely allocated str.
    Only for values without tab / newline / backslash (COPY text escapes those) and no NULLs."""
    buf = io.StringIO()
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            cur.copy_expert(f"COPY ({sql}) TO STDOUT", buf)
    finally:
        conn.close()
    text = buf.getvalue()
    del buf
    assert "\\" not in text and "\t" not in text, "value needs COPY unescaping"
    return np.array(text.split("\n")[:-1], dtype=object)
