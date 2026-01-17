from typing import Iterator
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from manager.core.exceptions import AppError
from manager.core.config import settings
from typing import Optional

# TODO: add typings here


def get_connection(host: Optional[str] = None):
    """
    Create a new PostgreSQL connection.
    Caller is responsible for closing it.
    """
    try:
        print(settings.POSTGRES_HOST)
        return psycopg2.connect(
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=host if host else settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
        )
    except psycopg2.OperationalError as e:
        raise AppError("DB_CONNECTION_FAILED") from e


@contextmanager
def get_cursor(
    commit: bool = True,
    dict_cursor: bool = False,
    host: Optional[str] = None
) -> Iterator[psycopg2.extensions.cursor]:
    """
    Context-managed database cursor with automatic commit / rollback.

    Usage:
        with get_cursor() as cur:
            cur.execute(...)
    """
    conn = get_connection(host=host)
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
