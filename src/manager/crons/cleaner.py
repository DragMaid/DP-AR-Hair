from time import sleep
from internal.connect import get_cursor
from os import remove
# This is getting ran as root so no need for manager
from core.config import settings


def clear_expired_uploads():
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT file_path FROM uploads
            WHERE expires_at < NOW()
        """, ())

        # Delete the files first then delete the record
        rows = cur.fetchall()
        for row in rows:
            remove(row["file_path"])

        cur.execute("""
            DELETE FROM uploads
            WHERE expires_at < NOW()
        """, ())

        print(f"Removed {len(rows)} expired uploads")


if __name__ == "__main__":
    # TODO: add a way to handle multiple cron jobs
    # TODO: add retry and all mighty cleanup (remove files not mentioned)
    while True:
        clear_expired_uploads()
        sleep(settings.TMP_FILE_CLEANUP_MIN * 60)
