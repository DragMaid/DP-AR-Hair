\restrict rYL3exetc2NzC4q493ur585ufU7WmU3AAip6ggSWX8eHJ6g5LRBQf6cufFXruV3

-- Dumped from database version 16.11
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
    'failed'
);


--
-- Name: image_types; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.image_types AS ENUM (
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
-- Name: user_roles; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.user_roles AS ENUM (
    'worker',
    'admin'
);


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
-- Name: assignments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assignments (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    task_id uuid,
    worker_id uuid,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: images; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.images (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    file_path text NOT NULL,
    type public.image_types NOT NULL,
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
    created_at timestamp without time zone DEFAULT now(),
    completed_at timestamp without time zone
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
    completed_at,
        CASE status
            WHEN 'processing'::public.task_status THEN 1
            WHEN 'pending'::public.task_status THEN 2
            WHEN 'completed'::public.task_status THEN 3
            ELSE NULL::integer
        END AS status_rank
   FROM public.tasks;


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
-- Name: users unique_username; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT unique_username UNIQUE (username);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


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
-- PostgreSQL database dump complete
--

\unrestrict rYL3exetc2NzC4q493ur585ufU7WmU3AAip6ggSWX8eHJ6g5LRBQf6cufFXruV3


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20251231022416');
