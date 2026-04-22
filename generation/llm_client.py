"""
generation/llm_client.py

Calls Claude with optional conversation history for memory.
"""

import anthropic
import os
from generation.prompts import SYSTEM_PROMPT
from generation.prompt_assembler import assemble_user_message

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def generate_plan(
    profile: dict,
    chunks: list[dict],
    error_hint: str = None,
    conversation_history: list[dict] = None   # ← NEW
) -> str:
    """
    Call Claude and return the raw response text (JSON string).

    Args:
        profile:              user profile dict
        chunks:               list of retrieved paper chunk dicts
        error_hint:           previous error for retry passes
        conversation_history: list of {role, message} dicts from Supabase
                              — gives Claude memory of previous adjustments
    """
    user_message = assemble_user_message(profile, chunks, error_hint=error_hint)

    # ── Build messages array with history ──────────────────────
    messages = []

    if conversation_history:
        # Inject previous turns so Claude remembers what was already changed
        history_context = "\n=== PREVIOUS CONVERSATION ===\n"
        history_context += "The user has already made these adjustments to their plan:\n\n"
        for turn in conversation_history:
            role_label = "User" if turn["role"] == "user" else "You (FitAI)"
            history_context += f"{role_label}: {turn['message']}\n"
        history_context += "\n=== END PREVIOUS CONVERSATION ===\n"
        history_context += "Take these previous adjustments into account when generating the updated plan.\n\n"

        # Prepend history to the user message
        user_message = history_context + user_message

    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    return response.content[0].text