-- ============================================================================
-- S.A.M — complete Supabase schema, deployment-ready.
--
-- Run this ONCE in the Supabase SQL Editor before pointing the app at it with
-- DATABASE_BACKEND=supabase. Every statement is idempotent (create if not
-- exists / add column if not exists), so re-running it is safe and is the
-- intended way to bring an older project up to date.
--
-- This file supersedes supabase_main_schema.sql and
-- supabase_escalated_questions.sql, which were written in two phases and left
-- the phase-2 session-state columns unapplied. Those columns are NOT optional:
-- the two-pass flow stores the confirm-before-ticket state in them, so without
-- them every ticket offer breaks on the customer's "yes".
-- ============================================================================

-- ---------------------------------------------------------------- sessions
-- One row per conversation. awaiting/pending_ticket_* carry the state machine
-- between turns, which is why they live here rather than in memory: Render
-- restarts and redeploys must not drop a conversation mid-ticket-offer.
create table if not exists sessions (
    session_id                 text primary key,
    created_at                 timestamptz not null default now(),
    total_turns                integer not null default 0,
    resolved                   boolean not null default false,
    awaiting                   text,          -- 'clarification' | 'ticket_confirmation' | 'troubleshoot_given'
    pending_ticket_query       text,          -- the question a ticket would be raised for
    pending_ticket_similarity  real
);

-- Brings an existing project (created from the phase-1 schema) up to date.
alter table sessions add column if not exists awaiting text;
alter table sessions add column if not exists pending_ticket_query text;
alter table sessions add column if not exists pending_ticket_similarity real;

-- ---------------------------------------------------------------- messages
-- Full transcript. Read back on every follow-up turn by Pass 1 to resolve
-- pronouns against history, so this table is on the hot path, not just an
-- audit log - hence the session_id index below.
create table if not exists messages (
    id                  bigint generated always as identity primary key,
    session_id          text not null references sessions(session_id),
    turn_number         integer not null,
    query               text not null,
    response            text not null,
    retrieved_chunks    text,
    similarity_scores   text,
    confidence_level    text,
    is_unknown_question boolean not null default false,
    response_latency_ms integer,
    input_tokens        integer,
    output_tokens       integer,
    created_at          timestamptz not null default now()
);

-- ------------------------------------------------------- unknown_questions
-- Questions the FAQ could not answer. This is the backlog the team works from
-- to decide what to add to the sheet.
create table if not exists unknown_questions (
    id              bigint generated always as identity primary key,
    session_id      text not null,
    query           text not null,
    similarity_score real,
    reviewed        boolean not null default false,
    answer          text,
    created_at      timestamptz not null default now()
);

-- ---------------------------------------------------- escalated_questions
-- Raised only after the customer explicitly confirms they want a ticket.
create table if not exists escalated_questions (
    id              bigint generated always as identity primary key,
    question        text not null,
    criticality     text not null check (criticality in ('low', 'medium', 'high')),
    similarity_score real,
    session_id      text not null,
    turn_number     integer,
    reviewed        boolean not null default false,
    created_at      timestamptz not null default now()
);

-- ---------------------------------------------------------- evaluation_logs
create table if not exists evaluation_logs (
    id              bigint generated always as identity primary key,
    message_id      bigint not null references messages(id),
    metric_name     text not null,
    passed          boolean not null,
    score           real,
    detail          text
);

-- ---------------------------------------------------------------- indexes
create index if not exists idx_messages_session_id         on messages(session_id);
create index if not exists idx_evaluation_logs_message_id  on evaluation_logs(message_id);
create index if not exists idx_unknown_questions_reviewed  on unknown_questions(reviewed);
-- Added for deployment: the two review queues are read by "newest unreviewed
-- first", which is a sequential scan without these once the tables grow.
create index if not exists idx_escalated_reviewed_created  on escalated_questions(reviewed, created_at desc);
create index if not exists idx_unknown_created             on unknown_questions(created_at desc);
-- Session lookups by recency, for any admin/reporting view.
create index if not exists idx_sessions_created            on sessions(created_at desc);

-- ============================================================================
-- ROW LEVEL SECURITY
--
-- The backend talks to Supabase with the service-role key, which bypasses RLS.
-- Enabling RLS with no permissive policy therefore changes nothing for the app
-- while closing the anon/public key off completely - so if the anon key ever
-- leaks, or someone points a client library at this project, they get nothing.
--
-- These tables hold customer conversation transcripts. Leave this on.
-- ============================================================================
alter table sessions            enable row level security;
alter table messages            enable row level security;
alter table unknown_questions   enable row level security;
alter table escalated_questions enable row level security;
alter table evaluation_logs     enable row level security;
