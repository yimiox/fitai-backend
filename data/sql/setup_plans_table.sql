-- Run this once in Supabase SQL editor
CREATE TABLE plans (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL,           -- links to your users table
    profile     JSONB NOT NULL,          -- the profile used to generate
    plan        JSONB NOT NULL,          -- the full generated plan
    chunks_used JSONB,                   -- which paper chunks were retrieved
    created_at  TIMESTAMP DEFAULT NOW(),
    version     INTEGER DEFAULT 1        -- for tracking plan updates
);

-- Index for fast user lookups
CREATE INDEX idx_plans_user_id ON plans(user_id);
CREATE INDEX idx_plans_created_at ON plans(created_at DESC);