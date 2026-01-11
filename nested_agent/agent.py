"""
Nested Orchestration Pattern

Demonstrates a  "nested" ADK workflow combining:
1) Sequential orchestration (overall pipeline)
2) Parallel orchestration (fan-out analysis)
3) Loop orchestration (iterative refinement with exit tool)

Use case:
- User asks for a SQL query given a task.
- We run a PARALLEL "analysis swarm" to extract structured requirements and risks.
- We then generate SQL once.
- Then we run a LOOP to validate/refine until PASS, exiting via an exit tool.
- Finally we produce a short final answer.

"""

import asyncio
from typing import Optional

from dotenv import load_dotenv
from google.genai import types

from google.adk.agents import (
    LlmAgent,
    SequentialAgent,
    LoopAgent,
    ParallelAgent,
)
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
load_dotenv()

APP_NAME = "nested_orchestration_sql_app_v1"
USER_ID = "dev_user_01"
SESSION_ID = "nested_orchestration_sql_session_01"
GEMINI_MODEL = "gemini-2.0-flash"

# -----------------------------------------------------------------------------
# State contract (keys referenced via placeholders)
# -----------------------------------------------------------------------------
STATE_USER_TASK = "user_sql_task"

# Parallel outputs
STATE_SCHEMA_ASSUMPTIONS = "schema_assumptions"
STATE_EDGE_CASES = "edge_cases"
STATE_QUERY_PLAN_NOTES = "query_plan_notes"

# Generator / loop keys
STATE_CURRENT_SQL = "current_sql"
STATE_CRITICISM = "criticism"

# Final
STATE_FINAL_RESPONSE = "final_response"

COMPLETION_PHRASE = "PASS"


# -----------------------------------------------------------------------------
# Tool: exit_loop
# -----------------------------------------------------------------------------
def exit_loop(tool_context: ToolContext) -> dict:
    """Stops the loop/run early by escalating (doc-style loop exit)."""
    print(f"  ✅ [Tool Call] exit_loop triggered by {tool_context.agent_name} (PASS reached)")
    tool_context.actions.escalate = True
    return {}


# -----------------------------------------------------------------------------
# Utility: safe event printing (text + tool calls)
# -----------------------------------------------------------------------------
def print_event_safely(event) -> None:
    """Prints text parts, and logs tool calls; avoids warnings about non-text parts."""
    if not event.content or not event.content.parts:
        return

    for part in event.content.parts:
        text = getattr(part, "text", None)
        if text:
            print(f"[{event.author}]: {text}\n")
            continue

        fc = getattr(part, "function_call", None) or getattr(part, "tool_call", None)
        if fc:
            name = getattr(fc, "name", "<unknown_tool>")
            args = getattr(fc, "args", None) or getattr(fc, "arguments", None) or {}
            print(f"[{event.author}] TOOL_CALL: {name}({args})\n")


# =============================================================================
# 1) PARALLEL "ANALYSIS SWARM" (Fan-out)
# =============================================================================

schema_assessor = LlmAgent(
    name="SchemaAssessor",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "You help translate a natural language SQL task into concrete assumptions.\n\n"
        f"User SQL task:\n{{{STATE_USER_TASK}}}\n\n"
        "Output:\n"
        "- A short bullet list of assumed schema details (tables, columns) if missing.\n"
        "- If the task already specifies schema, confirm it.\n"
        "Rules: Output only bullets."
    ),
    output_key=STATE_SCHEMA_ASSUMPTIONS,
)

edge_case_finder = LlmAgent(
    name="EdgeCaseFinder",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "Identify edge cases and correctness pitfalls for the SQL task.\n\n"
        f"User SQL task:\n{{{STATE_USER_TASK}}}\n\n"
        "Examples: NULL handling, ties, ordering, aggregation mistakes, missing GROUP BY.\n"
        "Output: 3-6 bullets only."
    ),
    output_key=STATE_EDGE_CASES,
)

performance_analyst = LlmAgent(
    name="PerformanceAnalyst",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "Provide lightweight query plan/performance notes.\n\n"
        f"User SQL task:\n{{{STATE_USER_TASK}}}\n\n"
        "Output:\n"
        "- 2-4 bullets on indexes / ORDER BY + LIMIT / aggregation cost.\n"
        "Rules: bullets only, no SQL."
    ),
    output_key=STATE_QUERY_PLAN_NOTES,
)

analysis_swarm = ParallelAgent(
    name="AnalysisSwarm",
    sub_agents=[schema_assessor, edge_case_finder, performance_analyst],
)


# =============================================================================
# 2) SEQUENTIAL: Generate initial SQL (after parallel analysis)
# =============================================================================

initial_sql_generator = LlmAgent(
    name="InitialSQLGenerator",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "You are a SQL generator.\n\n"
        f"User SQL task:\n{{{STATE_USER_TASK}}}\n\n"
        "Useful analysis context:\n"
        f"- Schema assumptions:\n{{{STATE_SCHEMA_ASSUMPTIONS}}}\n\n"
        f"- Edge cases:\n{{{STATE_EDGE_CASES}}}\n\n"
        f"- Performance notes:\n{{{STATE_QUERY_PLAN_NOTES}}}\n\n"
        "Task:\n"
        "- Generate an initial SQL query that satisfies the user task.\n\n"
        "Output requirements:\n"
        "- Output ONLY the SQL\n"
        "- Single statement\n"
        "- No markdown, no explanation"
    ),
    output_key=STATE_CURRENT_SQL,
)


# =============================================================================
# 3) LOOP: Critic -> Refiner until PASS
# =============================================================================

sql_critic = LlmAgent(
    name="SQLCritic",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "You are a strict SQL validator.\n\n"
        f"User SQL task:\n{{{STATE_USER_TASK}}}\n\n"
        "Analysis context (do not restate, use to validate correctness):\n"
        f"- Schema assumptions:\n{{{STATE_SCHEMA_ASSUMPTIONS}}}\n\n"
        f"- Edge cases:\n{{{STATE_EDGE_CASES}}}\n\n"
        f"- Performance notes:\n{{{STATE_QUERY_PLAN_NOTES}}}\n\n"
        f"SQL Draft:\n{{{STATE_CURRENT_SQL}}}\n\n"
        "Decision:\n"
        f"- If SQL is valid and satisfies the task, output EXACTLY: {COMPLETION_PHRASE}\n"
        "- Otherwise output concise error details of what must change.\n\n"
        "Rules:\n"
        "- Output ONLY 'PASS' or error text\n"
        "- No markdown"
    ),
    output_key=STATE_CRITICISM,
)

sql_refiner = LlmAgent(
    name="SQLRefiner",
    model=GEMINI_MODEL,
    include_contents="none",
    tools=[exit_loop],
    instruction=(
        "You refine SQL using critique, or exit when complete.\n\n"
        f"User SQL task:\n{{{STATE_USER_TASK}}}\n\n"
        f"Current SQL:\n{{{STATE_CURRENT_SQL}}}\n\n"
        f"Critique:\n{{{STATE_CRITICISM}}}\n\n"
        "Actions:\n"
        f"- IF critique is EXACTLY '{COMPLETION_PHRASE}': call exit_loop, output no text.\n"
        "- ELSE: fix the SQL to address critique and output ONLY corrected SQL.\n\n"
        "Constraints:\n"
        "- Single statement\n"
        "- No markdown, no explanation"
    ),
    output_key=STATE_CURRENT_SQL,
)

validation_loop = LoopAgent(
    name="SQLValidationLoop",
    sub_agents=[sql_critic, sql_refiner],
    max_iterations=6,
)


# =============================================================================
# 4) Final summarizer (optional, to produce a clean response artifact)
# =============================================================================

final_responder = LlmAgent(
    name="FinalResponder",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=(
        "Produce the final answer for the user.\n\n"
        f"Final SQL:\n{{{STATE_CURRENT_SQL}}}\n\n"
        "Output requirements:\n"
        "- Output the SQL only (no markdown)\n"
        "- No extra commentary"
    ),
    output_key=STATE_FINAL_RESPONSE,
)


# =============================================================================
# 5) NESTED ORCHESTRATION: Sequential pipeline that contains Parallel + Loop
# =============================================================================

# Root must be named `root_agent` in doc-style setups
root_agent = SequentialAgent(
    name="NestedOrchestrationSQLPipeline",
    sub_agents=[
        analysis_swarm,          # (Parallel) fan-out analysis
        initial_sql_generator,   # (Sequential) initial generation
        validation_loop,         # (Loop) iterate until PASS (exit_loop)
        final_responder,         # (Sequential) final response artifact
    ],
    description="Nested workflow: Parallel analysis -> initial SQL -> loop refine -> final output.",
)


# =============================================================================
# Runner
# =============================================================================
async def main() -> None:
    print("=" * 72)
    print("Nested Orchestration: Sequential + Parallel + Loop (SQL Use Case)")
    print("=" * 72)

    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

    # Seed ALL keys referenced by placeholders to avoid KeyError during injection
    initial_state = {
        STATE_USER_TASK: (
            "Write SQL to select the top 5 customers by total purchase amount "
            "from a table named orders with columns customer_id and amount."
        ),
        # Parallel outputs
        STATE_SCHEMA_ASSUMPTIONS: "",
        STATE_EDGE_CASES: "",
        STATE_QUERY_PLAN_NOTES: "",
        # SQL / loop keys
        STATE_CURRENT_SQL: "",
        STATE_CRITICISM: "",
        # Final
        STATE_FINAL_RESPONSE: "",
    }

    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )

    # Optional user message (doc-style relies on state; included for traceability)
    msg = types.Content(role="user", parts=[types.Part(text="Please generate the SQL using the nested workflow.")])

    print("\n🚀 Running...\n")

    final_text: Optional[str] = None

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=msg,
    ):
        print_event_safely(event)

        if event.is_final_response() and event.content and event.content.parts:
            final_text = getattr(event.content.parts[0], "text", None)

    updated = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
    )

    print("\n" + "-" * 72)
    print("✅ DONE")
    print("Final SQL (state['current_sql']):\n")
    print(updated.state.get(STATE_CURRENT_SQL, ""))

    print("\nFinal responder output (state['final_response']):\n")
    print(updated.state.get(STATE_FINAL_RESPONSE, ""))

    print("\nCritic status (state['criticism']):")
    print(updated.state.get(STATE_CRITICISM, ""))
    print("-" * 72)

    if final_text:
        print("\nFinal response event text:\n")
        print(final_text)


if __name__ == "__main__":
    asyncio.run(main())
