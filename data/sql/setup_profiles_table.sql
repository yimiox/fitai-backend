-- ============================================================
-- FitAI — Stage 3: User Profiles Table
-- Run this in Supabase SQL Editor after setup_db.sql
-- ============================================================

-- Stores every submitted questionnaire + all computed values
create table if not exists user_profiles (
  id           uuid primary key default gen_random_uuid(),

  -- Raw questionnaire answers (full JSON)
  raw_data     jsonb not null,

  -- Computed values: BMI, TDEE, macros, budget tier, etc.
  computed     jsonb not null,

  -- Pre-built RAG queries for Stage 4 retrieval
  rag_queries  text[] not null default '{}',

  -- Formatted text block for LLM prompt injection (Stage 4)
  llm_context  text,

  -- Plan generation status
  plan_status  text default 'pending',  -- 'pending' | 'generating' | 'ready' | 'error'
  plan_id      uuid,                    -- FK to plans table (Stage 5)

  created_at   timestamptz default now()
);

-- Index for quick lookups by plan status
create index if not exists user_profiles_status_idx
  on user_profiles (plan_status, created_at desc);

-- RLS
alter table user_profiles enable row level security;

create policy "service role full access on profiles"
  on user_profiles for all
  using (true);


-- ============================================================
-- Useful queries for development / debugging
-- ============================================================

-- See all profiles with their computed BMI + goal
-- select
--   id,
--   computed->>'bmi' as bmi,
--   computed->>'bmi_category' as bmi_cat,
--   computed->>'goal_label' as goal,
--   computed->>'budget_tier' as budget,
--   created_at
-- from user_profiles
-- order by created_at desc
-- limit 20;

-- Check a specific profile's LLM context
-- select llm_context from user_profiles where id = 'your-uuid-here';
