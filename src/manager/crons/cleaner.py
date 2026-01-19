from time import sleep
from threading import Thread
from os import remove
from manager.core.config import settings
from manager.internal.connect import get_cursor
import schedule


def wait_for_migration():
    while True:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'uploads'
                  AND column_name IN ('file_path', 'expires_at');
            """, ())
            if cur.fetchone():
                return
        sleep(1)


def clear_expired_uploads():
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT file_path
            FROM uploads
            WHERE expires_at < NOW()
        """, ())

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

        print(f"[Uploads] Removed {len(rows)} expired uploads")


def clear_timedout_assignments():
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            WITH expired_assignments AS (
                DELETE FROM assignments
                WHERE expires_at < NOW()
                RETURNING task_id, worker_id
            ),

            history_insert AS (
                INSERT INTO assignment_history (
                    task_id,
                    worker_id,
                    status,
                    log
                )
                SELECT
                    task_id,
                    worker_id,
                    'timeout'::assignment_status,
                    'Assignment timed out'
                FROM expired_assignments
            )

            UPDATE tasks t
            SET status = 'pending'
            WHERE t.id IN (
                SELECT DISTINCT task_id
                FROM expired_assignments
            )
            AND t.status = 'processing';
        """, ())
        print("[Assignments] Cleared timed-out assignments")


def run_job_in_thread(job_func):
    """Run a job in a separate thread to avoid blocking the scheduler"""
    Thread(target=job_func).start()


if __name__ == "__main__":
    print("Waiting for migration to be ran ...")
    wait_for_migration()

    print("Migration satisfied. Starting cron loop.")

    # Schedule multiple cron jobs with different timings
    schedule.every(settings.TMP_FILE_CLEANUP_MIN).minutes.do(
        run_job_in_thread, clear_expired_uploads)

    schedule.every(settings.TIMEDOUT_ASSIGNMENT_CLEANUP_MIN).minutes.do(
        run_job_in_thread, clear_timedout_assignments)

    while True:
        schedule.run_pending()
        sleep(1)
