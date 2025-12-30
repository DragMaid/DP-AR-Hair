CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    driving_image_id INT REFERENCES images(id) ON DELETE CASCADE,
    reference_image_id INT REFERENCES images(id) ON DELETE CASCADE,
    result_path TEXT NOT NULL,
    retry_count INT DEFAULT 0,
    priority INT DEFAULT 0,
    status processing_status DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);


CREATE TABLE workers (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE assignments (
    id SERIAL PRIMARY KEY,
    task_id INT REFERENCES tasks(id) ON DELETE CASCADE,
    worker_id INT REFERENCES workers(id) ON DELETE CASCADE,
    logs TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
);

CREATE INDEX idx_task_id ON tasks USING btree(id);
CREATE INDEX idx_task_status ON tasks USING btree(status);
CREATE INDEX idx_task_priority ON tasks USING btree(priority DESC);
CREATE INDEX idx_assingmnets_log USING gin(logs);
