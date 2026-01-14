-- migrate:up

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_cron;

GRANT USAGE ON SCHEMA cron TO CURRENT_USER;


CREATE TYPE user_roles as ENUM (
    'worker',
    'admin'
);

CREATE TYPE task_status AS ENUM (
    'pending',
    'processing',
    'completed'
);

CREATE TYPE assignment_status as ENUM (
    'succeed',
    'failed',
    'terminated',
    'timeout'
);

CREATE TYPE image_categories as ENUM (
    'driving',
    'reference',
    'generated'
);

CREATE TYPE upload_status as ENUM (
    'pending',
    'processed'
);

CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    category image_categories NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_image_path UNIQUE (file_path)
);

CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    driving_image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    reference_image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    result_path TEXT NOT NULL,
    retry_count INT DEFAULT 0,
    priority INT DEFAULT 0,
    status task_status DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_task_image_combination UNIQUE (driving_image_id, reference_image_id),
    CONSTRAINT unique_task_path UNIQUE (result_path)
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role user_roles NOT NULL, 
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_username UNIQUE (username)
);

CREATE TABLE ownership (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    admin_id UUID REFERENCES users(id) ON DELETE CASCADE,
    worker_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_ownership UNIQUE (admin_id, worker_id)
);

CREATE TABLE assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    worker_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    CONSTRAINT unique_assignment_task_worker UNIQUE (task_id, worker_id)
);

CREATE TABLE uploads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worker_id UUID REFERENCES users(id) ON DELETE CASCADE,
    assignment_id UUID REFERENCES assignments(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    status upload_status NOT NULL,
    category image_categories NOT NULL,
    expires_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_upload_path UNIQUE (file_path)
);

CREATE TABLE assignment_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    worker_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status assignment_status NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    log TEXT
);

CREATE VIEW tasks_ordered AS
SELECT *,
    CASE status
        WHEN 'processing' THEN 1
        WHEN 'pending'    THEN 2
        WHEN 'completed'  THEN 3
    END AS status_rank
FROM tasks;

CREATE VIEW assignment_history_ordered AS
SELECT *,
    CASE status
        WHEN 'failed'     THEN 1
        WHEN 'timeout'    THEN 2
        WHEN 'terminated' THEN 3
        WHEN 'succeed'    THEN 4
    END AS assignment_history_rank
FROM assignment_history;

CREATE INDEX idx_task_id ON tasks USING btree(id);
CREATE INDEX idx_upload_id ON uploads USING btree(id);
CREATE INDEX idx_task_status ON tasks USING btree(status);
CREATE INDEX idx_task_priority ON tasks USING btree(priority DESC);
CREATE INDEX idx_users_role ON users USING btree(role);
CREATE INDEX idx_ownership_worker_id ON ownership USING btree(worker_id);

-- Procedures
CREATE OR REPLACE PROCEDURE timeout_assignments()
LANGUAGE plpgsql
AS $$
BEGIN
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
    SET
        status = 'pending',
        retry_count = retry_count + 1
    WHERE t.id IN (
        SELECT DISTINCT task_id
        FROM expired_assignments
    )
    AND t.status = 'processing';

END;
$$;

-- Cron jobs
SELECT cron.schedule(
    'assignment-periodic-timeout',
    '*/5 * * * *',
    'CALL timeout_assignments()'
);

-- migrate:down
