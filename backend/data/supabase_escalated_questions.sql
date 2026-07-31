create table if not exists escalated_questions (
    id bigint generated always as identity primary key,
    question text not null,
    criticality text not null check (criticality in ('low', 'medium', 'high')),
    similarity_score real,
    session_id text not null,
    turn_number integer,
    reviewed boolean not null default false,
    created_at timestamptz not null default now()
);
