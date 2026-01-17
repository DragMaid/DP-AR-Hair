from time import sleep
from os import remove
from manager.core.config import settings
from manager.internal.connect import get_cursor


def wait_for_migration():
    while True:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'uploads'
                  AND column_name IN ('file_path', 'expires_at');
                """, ()
            )
            if cur.fetchone():
                return
        sleep(1)


def clear_expired_uploads():
    with get_cursor(dict_cursor=True) as cur:
        cur.execute(
            """
            SELECT file_path
            FROM uploads
            WHERE expires_at < NOW()
            """
        )

        rows = cur.fetchall()

        for row in rows:
            try:
                remove(row["file_path"])
            except FileNotFoundError:
                pass

        cur.execute(
            """
            DELETE FROM uploads
            WHERE expires_at < NOW()
            """
        )

        print(f"Removed {len(rows)} expired uploads")


if __name__ == "__main__":
    # TODO: add a way to handle multiple cron jobs
    # TODO: add retry and all mighty cleanup (remove files not mentioned)
    print("Waiting for migration to be ran ...")
    wait_for_migration()

    print("Migration satisfied. Starting cron loop.")

    while True:
        clear_expired_uploads()
        sleep(settings.TMP_FILE_CLEANUP_MIN * 60)
