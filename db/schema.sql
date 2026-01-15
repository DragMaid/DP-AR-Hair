\restrict 4GwbCO0DuPHUV06Tks8eg2pGO4kNQvUsu7cpPZndFJk5RvHLuCAZWNxiuGS7gqO

-- Dumped from database version 16.11 (Debian 16.11-1.pgdg12+1)
-- Dumped by pg_dump version 17.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: pg_cron; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION pg_cron; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pg_cron IS 'Job scheduler for PostgreSQL';


--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: assignment_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.assignment_status AS ENUM (
    'succeed',
    'failed',
    'terminated',
    'timeout'
);


--
-- Name: image_categories; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.image_categories AS ENUM (
    'driving',
    'reference',
    'generated'
);


--
-- Name: task_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.task_status AS ENUM (
    'pending',
    'processing',
    'completed'
);


--
-- Name: upload_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.upload_status AS ENUM (
    'pending',
    'processed'
);


--
-- Name: user_roles; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.user_roles AS ENUM (
    'worker',
    'admin'
);


--
-- Name: timeout_assignments(); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.timeout_assignments()
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


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: assignment_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assignment_history (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    task_id uuid,
    worker_id uuid,
    status public.assignment_status NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    log text
);


--
-- Name: assignment_history_ordered; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.assignment_history_ordered AS
 SELECT id,
    task_id,
    worker_id,
    status,
    created_at,
    log,
        CASE status
            WHEN 'failed'::public.assignment_status THEN 1
            WHEN 'timeout'::public.assignment_status THEN 2
            WHEN 'terminated'::public.assignment_status THEN 3
            WHEN 'succeed'::public.assignment_status THEN 4
            ELSE NULL::integer
        END AS assignment_history_rank
   FROM public.assignment_history;


--
-- Name: assignments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assignments (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    task_id uuid,
    worker_id uuid,
    created_at timestamp without time zone DEFAULT now(),
    expires_at timestamp without time zone
);


--
-- Name: images; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.images (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    file_path text NOT NULL,
    category public.image_categories NOT NULL,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: ownership; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ownership (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    admin_id uuid,
    worker_id uuid,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: tasks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tasks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    driving_image_id uuid,
    reference_image_id uuid,
    result_path text NOT NULL,
    retry_count integer DEFAULT 0,
    priority integer DEFAULT 0,
    status public.task_status DEFAULT 'pending'::public.task_status,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: tasks_ordered; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.tasks_ordered AS
 SELECT id,
    driving_image_id,
    reference_image_id,
    result_path,
    retry_count,
    priority,
    status,
    created_at,
        CASE status
            WHEN 'processing'::public.task_status THEN 1
            WHEN 'pending'::public.task_status THEN 2
            WHEN 'completed'::public.task_status THEN 3
            ELSE NULL::integer
        END AS status_rank
   FROM public.tasks;


--
-- Name: uploads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.uploads (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    worker_id uuid,
    assignment_id uuid,
    file_path text NOT NULL,
    status public.upload_status NOT NULL,
    category public.image_categories NOT NULL,
    expires_at timestamp without time zone DEFAULT now(),
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    username text NOT NULL,
    password_hash text NOT NULL,
    role public.user_roles NOT NULL,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: assignment_history assignment_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignment_history
    ADD CONSTRAINT assignment_history_pkey PRIMARY KEY (id);


--
-- Name: assignments assignments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignments
    ADD CONSTRAINT assignments_pkey PRIMARY KEY (id);


--
-- Name: images images_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images
    ADD CONSTRAINT images_pkey PRIMARY KEY (id);


--
-- Name: ownership ownership_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ownership
    ADD CONSTRAINT ownership_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: tasks tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);


--
-- Name: assignments unique_assignment_task_worker; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignments
    ADD CONSTRAINT unique_assignment_task_worker UNIQUE (task_id, worker_id);


--
-- Name: images unique_image_path; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images
    ADD CONSTRAINT unique_image_path UNIQUE (file_path);


--
-- Name: ownership unique_ownership; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ownership
    ADD CONSTRAINT unique_ownership UNIQUE (admin_id, worker_id);


--
-- Name: tasks unique_task_image_combination; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT unique_task_image_combination UNIQUE (driving_image_id, reference_image_id);


--
-- Name: tasks unique_task_path; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT unique_task_path UNIQUE (result_path);


--
-- Name: uploads unique_upload_path; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uploads
    ADD CONSTRAINT unique_upload_path UNIQUE (file_path);


--
-- Name: users unique_username; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT unique_username UNIQUE (username);


--
-- Name: uploads uploads_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uploads
    ADD CONSTRAINT uploads_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: idx_ownership_worker_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ownership_worker_id ON public.ownership USING btree (worker_id);


--
-- Name: idx_task_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_task_id ON public.tasks USING btree (id);


--
-- Name: idx_task_priority; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_task_priority ON public.tasks USING btree (priority DESC);


--
-- Name: idx_task_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_task_status ON public.tasks USING btree (status);


--
-- Name: idx_upload_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_upload_id ON public.uploads USING btree (id);


--
-- Name: idx_users_role; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_users_role ON public.users USING btree (role);


--
-- Name: assignment_history assignment_history_task_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignment_history
    ADD CONSTRAINT assignment_history_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.tasks(id) ON DELETE CASCADE;


--
-- Name: assignment_history assignment_history_worker_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignment_history
    ADD CONSTRAINT assignment_history_worker_id_fkey FOREIGN KEY (worker_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: assignments assignments_task_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignments
    ADD CONSTRAINT assignments_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.tasks(id) ON DELETE CASCADE;


--
-- Name: assignments assignments_worker_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assignments
    ADD CONSTRAINT assignments_worker_id_fkey FOREIGN KEY (worker_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: ownership ownership_admin_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ownership
    ADD CONSTRAINT ownership_admin_id_fkey FOREIGN KEY (admin_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: ownership ownership_worker_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ownership
    ADD CONSTRAINT ownership_worker_id_fkey FOREIGN KEY (worker_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: tasks tasks_driving_image_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_driving_image_id_fkey FOREIGN KEY (driving_image_id) REFERENCES public.images(id) ON DELETE CASCADE;


--
-- Name: tasks tasks_reference_image_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_reference_image_id_fkey FOREIGN KEY (reference_image_id) REFERENCES public.images(id) ON DELETE CASCADE;


--
-- Name: uploads uploads_assignment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uploads
    ADD CONSTRAINT uploads_assignment_id_fkey FOREIGN KEY (assignment_id) REFERENCES public.assignments(id) ON DELETE CASCADE;


--
-- Name: uploads uploads_worker_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uploads
    ADD CONSTRAINT uploads_worker_id_fkey FOREIGN KEY (worker_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict 4GwbCO0DuPHUV06Tks8eg2pGO4kNQvUsu7cpPZndFJk5RvHLuCAZWNxiuGS7gqO


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20251231022416');
