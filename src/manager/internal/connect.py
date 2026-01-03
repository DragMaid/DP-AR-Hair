from typing import Iterator
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from contextlib import contextmanager
from core.exceptions import AppError

load_dotenv()


def get_connection():
    """
    Create a new PostgreSQL connection.
    Caller is responsible for closing it.
    """
    try:
        return psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
        )
    except psycopg2.OperationalError as e:
        raise AppError("DB_CONNECTION_FAILED") from e


@contextmanager
def get_cursor(
    commit: bool = True,
    dict_cursor: bool = False,
) -> Iterator[psycopg2.extensions.cursor]:
    """
    Context-managed database cursor with automatic commit / rollback.

    Usage:
        with get_cursor() as cur:
            cur.execute(...)
    """
    conn = get_connection()
    cur = None
    try:
        cursor_factory = (
            psycopg2.extras.RealDictCursor if dict_cursor else None
        )
        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur
        if commit:
            conn.commit()
    except psycopg2.DatabaseError as e:
        conn.rollback()
        raise AppError("DB_QUERY_FAILED") from e
    finally:
        if cur is not None:
            cur.close()
        conn.close()
