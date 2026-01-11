import asyncio
from dotenv import load_dotenv

from google import adk
from google.adk.apps import App
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ============================================================
# 1) Define PARALLEL workers (Fan-out)
#    Each writes its output into session.state via output_key
# ============================================================

security_scanner = LlmAgent(
    name="SecurityAuditor",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a security auditor for code reviews.\n"
        "Given the code in the conversation, check for vulnerabilities such as:\n"
        "- prompt injection / unsafe tool usage patterns\n"
        "- secrets leakage / hard-coded tokens\n"
        "- unsafe deserialization\n"
        "- insecure defaults\n\n"
        "Return a short SECURITY report with:\n"
        "1) Findings (bullets)\n"
        "2) Risk level (Low/Med/High)\n"
        "3) Recommended fixes (bullets)\n"
    ),
    output_key="security_report",
)

style_checker = LlmAgent(
    name="StyleEnforcer",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a Python style reviewer.\n"
        "Check for PEP8 compliance and formatting issues.\n"
        "Return a short STYLE report with:\n"
        "1) Major issues (bullets)\n"
        "2) Minor issues (bullets)\n"
        "3) Suggested cleanups (bullets)\n"
    ),
    output_key="style_report",
)

complexity_analyzer = LlmAgent(
    name="PerformanceAnalyst",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a performance analyst.\n"
        "Review the code and comment on:\n"
        "- time complexity of key operations\n"
        "- potential bottlenecks\n"
        "- memory usage concerns\n"
        "- async/concurrency correctness\n\n"
        "Return a short PERFORMANCE report with:\n"
        "1) Complexity & bottlenecks (bullets)\n"
        "2) Performance risks (bullets)\n"
        "3) Improvements (bullets)\n"
    ),
    output_key="performance_report",
)

# Fan-out swarm: these three agents run concurrently
parallel_reviews = ParallelAgent(
    name="CodeReviewSwarm",
    sub_agents=[security_scanner, style_checker, complexity_analyzer],
)

# ============================================================
# 2) Gather/Synthesize step (Fan-in)
#    Reads {security_report}, {style_report}, {performance_report}
# ============================================================

pr_summarizer = LlmAgent(
    name="PRSummarizer",
    model="gemini-2.0-flash-exp",
    instruction=(
        "Create a consolidated Pull Request review.\n\n"
        "SECURITY REPORT:\n{security_report}\n\n"
        "STYLE REPORT:\n{style_report}\n\n"
        "PERFORMANCE REPORT:\n{performance_report}\n\n"
        "Output format:\n"
        "## Summary\n"
        "## Must-fix (blocking)\n"
        "## Should-fix (non-blocking)\n"
        "## Nice-to-have\n"
        "## Suggested next steps\n"
    ),
    output_key="final_review",
)

# ============================================================
# 3) Wrap parallel + summarizer into a single Sequential workflow
# ============================================================

workflow = SequentialAgent(
    name="PullRequestReviewWorkflow",
    sub_agents=[parallel_reviews, pr_summarizer],
)

# ============================================================
# 4) Run it end-to-end in ONE Runner call
# ============================================================

async def main():
    print("=" * 70)
    print("ParallelAgent (fan-out) + SequentialAgent (fan-in) Code Review Demo")
    print("=" * 70)

    app_name = "code_review_app"
    user_id = "user-123"

    # Create the App with the full workflow as root agent
    app = App(name=app_name, root_agent=workflow)

    # InMemory sessions are keyed by app_name + user_id + session_id.
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=app_name, user_id=user_id)

    # ---- IMPORTANT (based on your earlier KeyError issues) ----
    # Seed the state keys that pr_summarizer references via {…}.
    # If any are missing at instruction-injection time, ADK can throw KeyError.
    session.state["security_report"] = ""
    session.state["style_report"] = ""
    session.state["performance_report"] = ""
    session.state["final_review"] = ""

    runner = adk.Runner(app=app, session_service=session_service)

    # Example “PR diff” / code text.
    # In real usage, replace this with your actual code or diff.
    code_under_review = """
import os

API_KEY = "hardcoded-secret"

def do_eval(user_input):
    # naive prompt concatenation
    prompt = "Do something with: " + user_input
    return prompt
"""

    # We send the "code under review" as a user message so the agents can read it.
    msg = types.Content(role="user", parts=[types.Part(text=code_under_review)])

    print("\n🚀 Running workflow...\n")

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=msg,
    ):
        # Stream any text parts from agents
        if event.content and event.content.parts:
            t = getattr(event.content.parts[0], "text", None)
            if t:
                print(f"[{event.author}]: {t}\n")

        # When the final answer is emitted, show it clearly
        if event.is_final_response():
            final_text = event.content.parts[0].text if event.content.parts else ""
            print("\n✅ FINAL CONSOLIDATED REVIEW:\n")
            print(final_text)

if __name__ == "__main__":
    asyncio.run(main())
