-- migrate:up

CREATE TYPE task_status AS ENUM (
    'pending',
    'processing',
    'completed'
);

CREATE TYPE assignment_status as ENUM (
    'succeed',
    'processing',
    'failed'
);

CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (file_path)
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
    UNIQUE (result_path)
);


CREATE TABLE workers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (email)
);

CREATE TABLE assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    worker_id UUID REFERENCES workers(id) ON DELETE CASCADE,
    status assignment_status DEFAULT 'processing',
    logs TEXT, -- Is this best practice ? 
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE VIEW tasks_ordered AS
SELECT *,
    CASE status
        WHEN 'processing' THEN 1
        WHEN 'pending'    THEN 2
        WHEN 'completed'  THEN 3
    END AS status_rank
FROM tasks;

CREATE VIEW assignments_ordered AS
SELECT *,
    CASE status
        WHEN 'processing' THEN 1
        WHEN 'failed'     THEN 2
        WHEN 'succeed'    THEN 3
    END AS status_rank
FROM assignments;

CREATE INDEX idx_task_id ON tasks USING btree(id);
CREATE INDEX idx_task_status ON tasks USING btree(status);
CREATE INDEX idx_task_priority ON tasks USING btree(priority DESC);

-- migrate:down
