"""
generation/llm_client.py
 
Thin wrapper around the Anthropic SDK.
Called by pipeline.py — do not put FastAPI routes here.
"""
 
import anthropic
import os
from generation.prompts import SYSTEM_PROMPT
from generation.prompt_assembler import assemble_user_message
 
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
 
 
def generate_plan(profile: dict, chunks: list[dict], error_hint: str = None) -> str:
    """
    Call Claude and return the raw response text (JSON string).
 
    Args:
        profile:    user profile dict
        chunks:     list of retrieved paper chunk dicts
        error_hint: if this is a retry, pass the previous error so Claude can fix it
    """
    user_message = assemble_user_message(profile, chunks, error_hint=error_hint)
 
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
 
    return response.content[0].text