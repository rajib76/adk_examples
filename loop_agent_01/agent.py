"""
This script demonstrates the Loop orchestration pattern:

    1) Initial generator runs ONCE to create the first SQL draft.
    2) LoopAgent runs a Critic -> Refiner cycle for up to N iterations.
    3) When the Critic decides the SQL is correct, the Refiner calls an `exit_loop` tool.
    4) The tool sets `tool_context.actions.escalate = True`, which stops the loop/run early.

"""

import asyncio
from typing import Optional

from dotenv import load_dotenv
from google.genai import types

from google.adk.agents import LoopAgent, LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext

# -----------------------------------------------------------------------------
# Environment / runtime setup
# -----------------------------------------------------------------------------
# Loads .env so the Google / Gemini credentials and project configuration
# required by ADK can be picked up at runtime.
load_dotenv()

# -----------------------------------------------------------------------------
# Application identifiers
# -----------------------------------------------------------------------------
# These identifiers are used by ADK's session service to locate and store sessions.
# In production you generally:
# - set APP_NAME to a stable service identifier
# - set USER_ID per authenticated user (or service account)
# - set SESSION_ID per conversation / workflow instance
APP_NAME = "sql_loop_app_v1"
USER_ID = "dev_user_01"
SESSION_ID = "sql_loop_session_01"

# Model choice. Keep this configurable (env or config file) in production.
GEMINI_MODEL = "gemini-2.0-flash"

# -----------------------------------------------------------------------------
# Session state contract (keys used in prompts)
# -----------------------------------------------------------------------------
# IMPORTANT:
# ADK injects placeholders (e.g., {sql_task}) by reading session.state before
# each model call. Missing keys cause KeyError at runtime. Therefore we define
# a clear "state contract" and seed it on session creation.

STATE_SQL_TASK = "sql_task"         # Input: the user's SQL requirement/specification
STATE_CURRENT_SQL = "current_sql"   # Working draft: latest SQL produced by generator/refiner
STATE_CRITICISM = "criticism"       # Critic output: either "PASS" or error details

# Critic must output this exact string to signal completion.
COMPLETION_PHRASE = "PASS"

# -----------------------------------------------------------------------------
# Tool: exit_loop
# -----------------------------------------------------------------------------
def exit_loop(tool_context: ToolContext) -> dict:
    """
    Exit tool called by the Refiner when the SQL is acceptable.

    ADK-specific behavior:
    - Setting tool_context.actions.escalate = True signals the runtime to stop
      the current agent execution early (used as a clean "break" from a loop).
    - Tools should return JSON-serializable values (dict/list/str/number/etc.).

    This tool intentionally returns an empty dict because the action is side-effect driven.
    """
    # Production logging would use a structured logger (json logs) rather than print().
    print(f"  ✅ [Tool Call] exit_loop triggered by {tool_context.agent_name} (completion reached)")
    tool_context.actions.escalate = True
    return {}


# -----------------------------------------------------------------------------
# Utility: robust event printing
# -----------------------------------------------------------------------------
def print_event_safely(event) -> None:
    """
    Prints streaming events without triggering ADK's "non-text parts" warning.

    Why the warning happens:
    - When the Refiner calls a tool, the model output includes a non-text "function_call" part.
    - Some convenience methods try to concatenate text parts and warn you about non-text parts.

    This helper:
    - Prints text parts only (clean console output).
    - Optionally prints tool call metadata if present (useful during debugging).
    """
    if not event.content or not event.content.parts:
        return

    for part in event.content.parts:
        # Most human-readable output is emitted as text parts.
        text = getattr(part, "text", None)
        if text:
            print(f"[{event.author}]: {text}\n")
            continue

        # Tool calls appear as function_call/tool_call parts depending on SDK build.
        fc = getattr(part, "function_call", None) or getattr(part, "tool_call", None)
        if fc:
            name = getattr(fc, "name", "<unknown_tool>")
            args = getattr(fc, "args", None) or getattr(fc, "arguments", None) or {}
            print(f"[{event.author}] TOOL_CALL: {name}({args})\n")


# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
# Agent 1: Initial SQL generator (runs exactly once).
# Writes SQL into session.state["current_sql"] via output_key.
initial_generator = LlmAgent(
    name="InitialSQLGenerator",
    model=GEMINI_MODEL,
    include_contents="none",  # Doc-style: rely on state placeholders, not full conversation history
    instruction=(
        "You are a SQL generator.\n"
        "Generate an initial SQL query that satisfies the task below.\n\n"
        f"SQL Task:\n{{{STATE_SQL_TASK}}}\n\n"
        "Output requirements:\n"
        "- Output ONLY the SQL.\n"
        "- Single statement.\n"
        "- No markdown.\n"
        "- No explanation.\n"
    ),
    output_key=STATE_CURRENT_SQL,
    description="Creates the initial SQL draft.",
)

# Agent 2: SQL critic (inside the loop).
# Writes either:
# - "PASS" (exactly) when the SQL is good, OR
# - actionable error text describing what's wrong
critic_in_loop = LlmAgent(
    name="SQLCritic",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "You are a strict SQL validator.\n\n"
        f"SQL Task:\n{{{STATE_SQL_TASK}}}\n\n"
        f"SQL Draft:\n{{{STATE_CURRENT_SQL}}}\n\n"
        "Decision:\n"
        f"- If the SQL is valid and satisfies the task, output EXACTLY: {COMPLETION_PHRASE}\n"
        "- Otherwise, output concise error details (what is wrong and what must change).\n\n"
        "Rules:\n"
        "- Output ONLY 'PASS' or error text.\n"
        "- No markdown.\n"
        "- Be specific (e.g., missing GROUP BY, wrong aggregation, wrong limit/order, "
        "missing table/column references).\n"
    ),
    output_key=STATE_CRITICISM,
    description="Validates SQL and returns PASS or error details.",
)

# Agent 3: SQL refiner (inside the loop).
# If Critic says PASS -> call exit_loop tool.
# Else -> refine SQL and overwrite session.state["current_sql"].
refiner_in_loop = LlmAgent(
    name="SQLRefiner",
    model=GEMINI_MODEL,
    include_contents="none",
    tools=[exit_loop],
    instruction=(
        "You are a SQL refiner.\n\n"
        f"SQL Task:\n{{{STATE_SQL_TASK}}}\n\n"
        f"Current SQL:\n{{{STATE_CURRENT_SQL}}}\n\n"
        f"Critique:\n{{{STATE_CRITICISM}}}\n\n"
        "Actions:\n"
        f"- IF the critique is EXACTLY '{COMPLETION_PHRASE}':\n"
        "  You MUST call the tool 'exit_loop'. Output no text.\n"
        "- ELSE:\n"
        "  Fix the SQL to address the critique AND satisfy the SQL Task.\n"
        "  Output ONLY the corrected SQL.\n\n"
        "Constraints:\n"
        "- Single SQL statement.\n"
        "- No markdown.\n"
        "- No explanation.\n"
    ),
    output_key=STATE_CURRENT_SQL,
    description="Refines SQL or exits via tool when completion is reached.",
)

# Loop agent: repeatedly runs Critic -> Refiner until:
# - exit_loop escalates (early stop), OR
# - max_iterations is hit (safety stop)
validation_loop = LoopAgent(
    name="SQLValidationLoop",
    sub_agents=[critic_in_loop, refiner_in_loop],
    max_iterations=6,
)

# Root workflow: initial generation once, then loop refinement.
# This mirrors the ADK docs structure where the "root agent" is a sequential pipeline.
root_agent = SequentialAgent(
    name="SQLIterativePipeline",
    sub_agents=[initial_generator, validation_loop],
    description="Generates SQL then iteratively validates/refines until PASS.",
)

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
async def main() -> None:
    """
    Entrypoint for running the workflow in-memory.

    In production, you would typically:
    - use a persistent session service (datastore/redis/db)
    - generate session IDs per request
    - implement structured logging/tracing
    - validate user input and enforce guardrails
    """
    print("=" * 70)
    print("LoopAgent Example (Doc-style) — SQL Generator/Critic")
    print("=" * 70)

    # InMemoryRunner provides:
    # - a runner to execute the agent graph
    # - an in-memory session service (runner.session_service)
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

    # Seed state keys so placeholder injection never KeyErrors.
    # This is essential because prompts reference these keys before the first model call.
    initial_state = {
        STATE_SQL_TASK: (
            "Write SQL to select the top 5 customers by total purchase amount "
            "from a table named orders with columns customer_id and amount."
        ),
        STATE_CURRENT_SQL: "",
        STATE_CRITICISM: "",
    }

    # Create a session bound to (app_name, user_id, session_id).
    # In production, the session_id should be unique per conversation/workflow run.
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )

    # A user message is optional in this doc-style pattern because include_contents='none'
    # and the agents rely primarily on session.state. We include it for traceability.
    msg = types.Content(role="user", parts=[types.Part(text="Generate and validate the SQL.")])

    print("\n🚀 Running...\n")

    # Some ADK builds emit a "final response" event; others may terminate via escalation.
    # We capture it if present, but we also always fetch the final SQL from session.state.
    final_text: Optional[str] = None

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=msg,
    ):
        # Print streamed output safely (text parts + tool calls).
        print_event_safely(event)

        # Capture final-response text (may be absent when the final action is a tool call).
        if event.is_final_response() and event.content and event.content.parts:
            final_text = getattr(event.content.parts[0], "text", None)

    # Always read final SQL from session state; this is the "source of truth" for this workflow.
    updated = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
    )

    print("\n" + "-" * 70)
    print("✅ DONE")
    print("Final SQL (from session.state):\n")
    print(updated.state.get(STATE_CURRENT_SQL, ""))

    print("\nFinal Critic status (state['criticism']):")
    print(updated.state.get(STATE_CRITICISM, ""))
    print("-" * 70)

    # Optional: also show final response event text if ADK emitted one.
    if final_text:
        print("\nFinal response event text:\n")
        print(final_text)


if __name__ == "__main__":
    asyncio.run(main())
