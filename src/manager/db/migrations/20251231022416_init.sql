-- migrate:up

CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
    'failed'
);

CREATE TYPE image_types as ENUM (
    'driving',
    'reference',
    'generated'
);

CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    type image_types NOT NULL,
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
    completed_at TIMESTAMP,
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

CREATE TABLE assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    worker_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_assignment_task_worker UNIQUE (task_id, worker_id)
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

CREATE INDEX idx_task_id ON tasks USING btree(id);
CREATE INDEX idx_task_status ON tasks USING btree(status);
CREATE INDEX idx_task_priority ON tasks USING btree(priority DESC);
CREATE INDEX idx_users_role ON users USING btree(role)

-- migrate:down
