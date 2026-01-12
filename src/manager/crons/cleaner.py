from time import sleep
from internal.connect import get_cursor
from os import remove

INTERVAL_MIN = 10


def clear_expired_uploads():
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            DELETE FROM uploads
            WHERE expires_at < NOW()
            RETURNING file_path
        """, ())

        rows = cur.fetchall()
        for row in rows:
            remove(row["file_path"])
        print(f"Removed {len(rows)} expired uploads")


if __name__ == "__main__":
    # TODO: add a way to handle multiple cron jobs
    # TODO: add retry and all mighty cleanup (remove files not mentioned)
    while True:
        clear_expired_uploads()
        sleep(INTERVAL_MIN * 60)
