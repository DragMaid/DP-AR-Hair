CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing',
    'completed',
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
    driving_image_id INT REFERENCES images(id) ON DELETE CASCADE,
    reference_image_id INT REFERENCES images(id) ON DELETE CASCADE,
    result_path TEXT NOT NULL,
    retry_count INT DEFAULT 0,
    priority INT DEFAULT 0,
    status processing_status DEFAULT 'pending',
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
    task_id INT REFERENCES tasks(id) ON DELETE CASCADE,
    worker_id INT REFERENCES workers(id) ON DELETE CASCADE,
    logs TEXT, -- Is this best practice ? 
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_task_id ON tasks USING btree(id);
CREATE INDEX idx_task_status ON tasks USING btree(status);
CREATE INDEX idx_task_priority ON tasks USING btree(priority DESC);
CREATE INDEX idx_assignment_log ON assignments USING gin(logs);
