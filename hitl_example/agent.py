"""
Human-in-the-Loop (YES/NO)

What this demo shows:
1) An Agent decides it needs to call a sensitive tool (e.g., "refund_order").
2) The tool is wrapped with FunctionTool(..., require_confirmation=True).
   ADK pauses before executing the tool and emits a FunctionCall named:
       adk_request_confirmation
3) We (a human) approve/deny in the terminal (y/n).
4) We resume by sending a FunctionResponse with:
   - id   = function_call_id from the confirmation request
   - name = "adk_request_confirmation"
   - response = {"confirmed": true/false}

"""

import asyncio
from typing import Optional, Tuple

from dotenv import load_dotenv
from google.genai import types

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
load_dotenv()

APP_NAME = "hitl_confirmation_demo"
USER_ID = "user_123"
SESSION_ID = "session_123"
MODEL = "gemini-2.0-flash"


# ---------------------------------------------------------------------
# A "sensitive" tool that we want human approval for
# NOTE: This tool will NOT run until the human confirms.
# ---------------------------------------------------------------------
def refund_order(order_id: str, amount_usd: float) -> dict:
    """
    Example sensitive operation.

    In production, this might:
    - call a payment processor
    - create an audit log entry
    - notify downstream systems

    Returns a JSON-serializable dict (recommended for ADK tools).
    """
    # This is a stubbed "side effect":
    return {
        "status": "refunded",
        "order_id": order_id,
        "amount_usd": amount_usd,
    }


# ---------------------------------------------------------------------
# Agent definition
# - The key is wrapping the tool with FunctionTool(require_confirmation=True)
# - ADK will pause to request yes/no approval before executing the tool.
# ---------------------------------------------------------------------
root_agent = Agent(
    name="root_agent",
    model=MODEL,
    instruction=(
        "You are a customer support assistant.\n"
        "If the user asks for a refund, call the refund_order tool with:\n"
        "- order_id (string)\n"
        "- amount_usd (number)\n"
        "Otherwise, answer normally.\n"
        "After the tool returns, summarize the outcome to the user."
    ),
    tools=[
        # This is the critical piece: enable boolean tool confirmation.
        FunctionTool(refund_order, require_confirmation=True),
    ],
)


# ---------------------------------------------------------------------
# Helpers: parse ADK events to detect the confirmation request
# ---------------------------------------------------------------------
def _extract_confirmation_call(event) -> Optional[Tuple[str, str]]:
    """
    If this event contains the special ADK confirmation FunctionCall, return:
      (function_call_id, function_name)

    ADK docs say:
      - function_call.name will be 'adk_request_confirmation'
      - function_response we send back must use name 'adk_request_confirmation'
      - function_response.id must match the function_call_id
    """
    if not event.content or not event.content.parts:
        return None

    for part in event.content.parts:
        fc = getattr(part, "function_call", None)
        if not fc:
            continue

        # The confirmation request function name is fixed by ADK.
        if getattr(fc, "name", None) == "adk_request_confirmation":
            function_call_id = getattr(fc, "id", None)
            if function_call_id:
                return (function_call_id, fc.name)

    return None


def _make_confirmation_response(function_call_id: str, confirmed: bool) -> types.Content:
    """
    Build the Content payload that "approves/denies" the tool call.

    Must follow ADK format:
      function_response.id   == function_call_id
      function_response.name == "adk_request_confirmation"
      function_response.response = {"confirmed": true/false}
    """
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=function_call_id,
                    name="adk_request_confirmation",
                    response={"confirmed": confirmed},
                )
            )
        ],
    )


# ---------------------------------------------------------------------
# Main: run -> pause -> prompt -> resume
# ---------------------------------------------------------------------
async def main():
    print("=" * 70)
    print("ADK Human-in-the-Loop (YES/NO) Tool Confirmation")
    print("=" * 70)

    # 1) Create an in-memory session service + runner
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # 2) Send a user message that will cause the agent to call the tool
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text="Please refund $19.99 for order A-1007")],
    )

    print("\n[PHASE 1] Running until confirmation is requested...\n")

    confirmation_call_id: Optional[str] = None

    # Run the agent. We stop once we detect the confirmation request.
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_msg,
    ):
        # Print any text parts for visibility
        if event.content and event.content.parts:
            for p in event.content.parts:
                t = getattr(p, "text", None)
                if t:
                    print(f"[{event.author}]: {t}")

        # Detect the built-in ADK confirmation request function call
        maybe = _extract_confirmation_call(event)
        if maybe:
            confirmation_call_id, _ = maybe
            print("\n--- PAUSED: tool confirmation required ---")
            print(f"function_call_id = {confirmation_call_id}")
            print("-----------------------------------------\n")
            break

        # If somehow we finish without needing confirmation, we can end here
        if event.is_final_response():
            print("\n✅ Completed without confirmation step.")
            return

    if not confirmation_call_id:
        print("No confirmation request was emitted. (Maybe the agent didn't call the tool.)")
        return

    # 3) Human decision (YES/NO)
    raw = input("Approve this tool execution? (y/n): ").strip().lower()
    approved = raw in {"y", "yes"}

    # 4) Resume by sending the FunctionResponse message
    #    This is the "remote confirmation" flow from the docs.
    confirm_msg = _make_confirmation_response(
        function_call_id=confirmation_call_id,
        confirmed=approved,
    )

    print("\n[PHASE 2] Resuming with confirmation response...\n")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=confirm_msg,
    ):
        if event.content and event.content.parts:
            for p in event.content.parts:
                t = getattr(p, "text", None)
                if t:
                    print(f"[{event.author}]: {t}")

        if event.is_final_response():
            print("\n✅ DONE")
            # The final answer is usually in the final response event text part
            if event.content and event.content.parts:
                final_text = getattr(event.content.parts[0], "text", None)
                if final_text:
                    print("\nFinal response:\n", final_text)
            break


if __name__ == "__main__":
    asyncio.run(main())
