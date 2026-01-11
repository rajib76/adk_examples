"""
Sequential Agent

This example demonstrates:
1. How SequentialAgent executes agents in strict order
2. How output_key writes values into shared session state
3. How to safely integrate tools without relying on LLM tool-calling
"""

import asyncio

# Load environment variables (GOOGLE_API_KEY, etc.)
from dotenv import load_dotenv

# Core ADK imports
from google import adk
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService

# Gemini content primitives
from google.genai import types

load_dotenv()

# ============================================================
# TOOL: Deterministic async function
# ============================================================
# This is a plain async Python function.
# ADK treats this as a "tool", but we DO NOT let the LLM decide
# when/how to call it. Instead, we call it ourselves inside
# a custom agent (DataFetcherAgent).
#
async def database_search(query: str, category: str) -> str:
    """
    Simulated database lookup.

    Args:
        query (str): The user's original request
        category (str): Classified intent (support/sales/technical)

    Returns:
        str: Mock database result
    """
    mock_results = {
        "support": (
            "Support ticket #12345: Login authentication issue "
            "- reset password at https://example.com/reset"
        ),
        "sales": (
            "Sales opportunity: Enterprise plan upgrade "
            "- contact sales@example.com"
        ),
        "technical": (
            "Technical docs: API authentication guide "
            "- see https://docs.example.com/auth"
        ),
    }

    # Return category-specific result or a fallback
    return mock_results.get(category, "No results found")


# ============================================================
# Helper: Extract the latest user message from the session
# ============================================================
# ADK stores all conversation turns as Events in session.events.
# This helper walks backward through those events and returns
# the most recent user-authored text message.
def _latest_user_text(session) -> str:
    """
    Retrieve the most recent user text message from the session.

    This is necessary because:
    - The DataFetcherAgent runs AFTER the classifier
    - We want the original user query, not the classifier output
    """
    for ev in reversed(session.events):
        if ev.author == "user" and ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "text", None):
                    return p.text
    return ""


# ============================================================
# AGENT 1: Intent Classifier (LLM-based)
# ============================================================
# This agent:
# - Reads the user message
# - Produces a single token classification
# - Writes its output into session.state["category"]
#
# The key idea: output_key == shared state write
classifier = LlmAgent(
    name="classifier",
    model="gemini-2.0-flash-exp",
    instruction=(
        "Classify the user's intent.\n"
        "Return ONLY one word: support, sales, or technical.\n\n"
        "Rules:\n"
        "- login/password/account access issues => support\n"
        "- buying/upgrading/pricing => sales\n"
        "- api/docs/integration/debugging => technical"
    ),
    # Whatever text the model produces becomes:
    # session.state["category"]
    output_key="category",
)


# ============================================================
# AGENT 2: Data Fetcher (Custom deterministic agent)
# ============================================================
# This agent is NOT an LlmAgent.
# It subclasses BaseAgent and performs logic directly.
#
# Why?
# - We want guaranteed execution
# - We want guaranteed state writes
# - We want to avoid fragile LLM tool-calling semantics
class DataFetcherAgent(BaseAgent):
    def __init__(self):
        # Every agent must have a name
        super().__init__(name="data_fetcher")

    async def _run_async_impl(self, ctx):
        """
        Core execution method for this agent.

        ctx (InvocationContext) gives access to:
        - ctx.session.state       → shared mutable state
        - ctx.session.events      → full conversation history
        - ctx.invocation_id       → tracing/debugging
        """

        # Read the classifier output from shared state
        category = ctx.session.state.get("category").strip()

        # Extract the original user request
        query = _latest_user_text(ctx.session)

        # Call the tool deterministically
        result = await database_search(
            query=query,
            category=category,
        )

        # We explicitly write into session.state.
        # This guarantees that {search_results} exists
        # for downstream instruction injection.
        ctx.session.state["search_results"] = result

        # Emit an Event so streaming logs show progress.
        # This is optional but very useful for debugging.
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=result)]
            ),
        )


# Instantiate the custom agent
data_fetcher = DataFetcherAgent()


# ============================================================
# AGENT 3: Final Responder (LLM-based)
# ============================================================
# This agent:
# - Reads values from session.state
# - Uses them via {category} and {search_results}
# - Produces the final user-facing response
#
# Because previous agents ALWAYS write these keys,
# instruction injection is safe and deterministic.
responder = LlmAgent(
    name="responder",
    model="gemini-2.0-flash-exp",
    instruction=(
        "Create a helpful response.\n\n"
        "**Category**: {category}\n"
        "**Database Info**: {search_results}\n\n"
        "Generate a direct answer addressing the user's issue."
    ),
    output_key="final_response",
)


# ============================================================
# SEQUENTIAL PIPELINE
# ============================================================
# Execution order is strictly enforced:
# 1. classifier
# 2. data_fetcher
# 3. responder
#
# Each agent runs AFTER the previous one completes.
customer_service_agent = SequentialAgent(
    name="customer_service",
    sub_agents=[
        classifier,
        data_fetcher,
        responder,
    ],
)


# ============================================================
# APPLICATION ENTRYPOINT
# ============================================================
async def main():
    print("=" * 60)
    print("Sequential Agent with output_key Example (Robust Fix)")
    print("=" * 60 + "\n")

    app_name = "customer_service"
    user_id = "user-123"

    # In-memory session store (perfect for local dev)
    session_service = InMemorySessionService()

    # Create a new session for this user
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    # Seed state keys so instruction injection
    # never fails even during partial runs.
    session.state["category"] = ""
    session.state["search_results"] = ""
    session.state["final_response"] = ""

    # Runner is the canonical execution engine in ADK.
    # It handles:
    # - adding user messages
    # - managing invocation context
    # - streaming events
    runner = adk.Runner(
        agent=customer_service_agent,
        app_name=app_name,
        session_service=session_service,
    )

    # User input
    user_text = "I can't log into my account"
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_text)],
    )

    print("🚀 Executing pipeline...\n")

    # Stream execution events
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        # Print intermediate outputs (classifier, data_fetcher)
        if event.content and event.content.parts:
            text = getattr(event.content.parts[0], "text", None)
            if text:
                print(f"[{event.author}]: {text}")

        # Detect and print the final response
        if event.is_final_response():
            print("\n✅ Final response:")
            print(event.content.parts[0].text)


# Standard asyncio entrypoint
if __name__ == "__main__":
    asyncio.run(main())
