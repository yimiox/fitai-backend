import anthropic
import os
from prompts import SYSTEM_PROMPT
from prompt_assembler import assemble_user_message

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def generate_plan(profile: dict, chunks: list[dict]) -> str:
    """
    Call Claude and return the raw response text (JSON string).
    """
    user_message = assemble_user_message(profile, chunks)

    response = client.messages.create(
        model="claude-sonnet-4-5",   # fast + smart, good for structured output
        max_tokens=4096,             # plans can be long — don't skimp here
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Extract the text content from the response
    raw_text = response.content[0].text

    return raw_text


# FastAPI endpoint that ties everything together
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query_builder import build_retrieval_queries
from app.services.retriever import retrieve_chunks  # your pgvector search function CHANGED from retreival

app = FastAPI()

class ProfileRequest(BaseModel):
    profile: dict

@app.post("/generate-plan")
async def generate_plan_endpoint(request: ProfileRequest):
    profile = request.profile

    # Step 1: build queries from profile
    queries = build_retrieval_queries(profile)

    # Step 2: retrieve relevant paper chunks
    chunks = retrieve_chunks(queries, top_k=8)

    # Step 3: generate plan with LLM
    raw_response = generate_plan(profile, chunks)

    # Step 4: parse + validate (next step)
    from parser import parse_and_validate
    plan = parse_and_validate(raw_response, chunks)

    return plan