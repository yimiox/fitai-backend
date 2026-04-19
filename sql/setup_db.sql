-- ============================================================
-- FitAI — Supabase Database Setup
-- Run this in your Supabase SQL Editor (one time only)
-- ============================================================

-- Step 1: Enable the pgvector extension
-- This adds vector similarity search to Postgres
create extension if not exists vector;


-- Step 2: Create the papers metadata table
-- Stores one row per uploaded research paper
create table if not exists papers (
  id          uuid primary key default gen_random_uuid(),
  title       text not null,
  authors     text,
  year        int,
  doi         text,
  domain_tags text[],          -- e.g. ['strength training', 'hypertrophy']
  file_name   text,
  created_at  timestamptz default now()
);


-- Step 3: Create the paper_chunks table
-- Stores every text chunk extracted from every paper
-- The embedding column holds the vector (1536 dims for OpenAI, 384 for local)
create table if not exists paper_chunks (
  id          uuid primary key default gen_random_uuid(),
  paper_id    uuid references papers(id) on delete cascade,
  paper_title text not null,
  domain_tags text[],
  chunk_index int not null,           -- position of this chunk within the paper
  chunk_text  text not null,
  token_count int,
  embedding   vector(1536),          -- change to vector(384) if using local model
  created_at  timestamptz default now()
);


-- Step 4: Create the IVFFlat index for fast similarity search
-- This makes cosine search fast even with thousands of chunks
-- NOTE: Only run AFTER you have loaded at least some data.
--       IVFFlat needs rows to build the index. Run this after first ingestion.

-- create index on paper_chunks
--   using ivfflat (embedding vector_cosine_ops)
--   with (lists = 100);

-- For now, an exact search index works fine for small libraries (< 10,000 chunks):
create index if not exists paper_chunks_embedding_idx
  on paper_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 10);


-- Step 5: Create the retrieval function
-- This is called from Python to find the most relevant chunks for a query
create or replace function match_chunks (
  query_embedding  vector(1536),
  match_count      int     default 6,
  domain_filter    text[]  default null    -- optional: filter by domain
)
returns table (
  id          uuid,
  paper_title text,
  domain_tags text[],
  chunk_text  text,
  similarity  float
)
language sql stable
as $$
  select
    pc.id,
    pc.paper_title,
    pc.domain_tags,
    pc.chunk_text,
    1 - (pc.embedding <=> query_embedding) as similarity
  from paper_chunks pc
  where
    domain_filter is null
    or pc.domain_tags && domain_filter       -- && = array overlap operator
  order by pc.embedding <=> query_embedding  -- <=> = cosine distance
  limit match_count;
$$;


-- Step 6: Row-level security (optional but good practice)
-- Allow read-only access to chunks from your backend service role
alter table papers       enable row level security;
alter table paper_chunks enable row level security;

-- Service role (your backend) can do everything
create policy "service role full access on papers"
  on papers for all
  using (true);

create policy "service role full access on chunks"
  on paper_chunks for all
  using (true);


-- ============================================================
-- Verification queries — run these to confirm setup worked
-- ============================================================

-- Check extensions
-- select * from pg_extension where extname = 'vector';

-- Check tables
-- select table_name from information_schema.tables
-- where table_schema = 'public';

-- Check the function exists
-- select routine_name from information_schema.routines
-- where routine_name = 'match_chunks';
