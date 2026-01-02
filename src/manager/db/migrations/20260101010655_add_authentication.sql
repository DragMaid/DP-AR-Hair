-- migrate:up

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE user_roles as ENUM (
    'worker',
    'admin'
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role user_roles NOT NULL, 
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (username)
);

INSERT INTO users (id, username, password_hash, role, created_at) 
SELECT
    w.id,
    w.email,
    crypt('nopassword', gen_salt('bf', 12)),
    'worker'::user_roles,
    w.created_at
FROM workers w;

ALTER TABLE assignments 
DROP CONSTRAINT assignments_worker_id_fkey;

ALTER TABLE assignments
ADD CONSTRAINT assignments_worker_id_fkey
FOREIGN KEY (worker_id) REFERENCES users(id) ON DELETE CASCADE;

DROP TABLE workers;

CREATE INDEX idx_users_role ON users USING btree(role)

-- migrate:down
