create table if not exists sessions (
    session_id      text primary key,
    created_at      timestamptz not null default now(),
    total_turns     integer not null default 0,
    resolved        boolean not null default false
);

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

create table if not exists unknown_questions (
    id              bigint generated always as identity primary key,
    session_id      text not null,
    query           text not null,
    similarity_score real,
    reviewed        boolean not null default false,
    answer          text,
    created_at      timestamptz not null default now()
);

create table if not exists evaluation_logs (
    id              bigint generated always as identity primary key,
    message_id      bigint not null references messages(id),
    metric_name     text not null,
    passed          boolean not null,
    score           real,
    detail          text
);

create index if not exists idx_messages_session_id on messages(session_id);
create index if not exists idx_evaluation_logs_message_id on evaluation_logs(message_id);
create index if not exists idx_unknown_questions_reviewed on unknown_questions(reviewed);
