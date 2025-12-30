DO $$ BEGIN
CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


CREATE TABLE images(
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_path TEXT NOT NULL,
    status processing_status NOT NULL DEFAULT 'pending',
    worker_id TEXT,
    locked_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    result_path TEXT,
    error_message TEXT,

    CONSTRAINT uq_image_path UNIQUE (image_path)
);

CREATE INDEX idx_image_status ON images(status);

