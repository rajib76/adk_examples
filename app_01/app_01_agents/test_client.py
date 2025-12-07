# This code has been taken from https://github.com/jackwotherspoon/currency-agent/blob/main/currency_agent/test_client.py
# I just modified the output to make the JSON print more readable and added more comments
import os
import traceback
from typing import Any
from uuid import uuid4
import httpx
# Import necessary classes from the Agent-to-Agent (a2a) library
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    SendMessageResponse,
    GetTaskResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
)

# Configuration: Load the Agent URL from environment variables or default to localhost
# load_dotenv() # Uncomment if using a .env file
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8001")


def create_send_message_payload(
        text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """
    Helper function to structure the message payload for the A2A API.

    Args:
        text: The content of the user's message.
        task_id: (Optional) ID of existing task, used for multi-turn conversations.
        context_id: (Optional) ID of the specific conversation context/turn.
    """
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": uuid4().hex,  # Generate a unique ID for this specific message
        },
    }
    # If replying to an ongoing task, include the specific identifiers
    if task_id:
        payload["message"]["taskId"] = task_id
    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


def print_json_response(response: Any, description: str) -> None:
    """Utility to pretty-print Pydantic models or JSON responses for easier debugging."""
    print(f"--- {description} ---")
    if hasattr(response, "root"):
        print(f"{response.root.model_dump_json(exclude_none=True, indent=2)}\n")
    else:
        print(f"{response.model_dump(mode='json', exclude_none=True, indent=2)}\n")


async def run_single_turn_test(client: A2AClient) -> None:
    """
    Scenario 1: Simple Request
    Tests sending a message and retrieving the task status.
    """
    # 1. Prepare the payload
    send_message_payload = create_send_message_payload(text="Where is TajMahal?")

    # 2. Wrap payload in a Request object including a request ID
    request = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )
    print("--- ✉️  Single Turn Request ---")

    # 3. Send message to the agent
    response: SendMessageResponse = await client.send_message(request)
    print_json_response(response, "📥 Single Turn Request Response")
    # 4. Validation: Ensure we got a success response
    if not isinstance(response.root, SendMessageSuccessResponse):
        print("received non-success response. Aborting get task ")
        return
    # 5. Validation: Ensure the result contains a Task object
    if not isinstance(response.root.result, Task):
        print("received non-task response. Aborting get task ")
        return
    # 6. Extract the Task ID from the response for querying
    task_id: str = response.root.result.id

    print("--- ❔ Query Task ---")
    # 7. Test the 'Get Task' functionality to retrieve task state by ID
    get_request = GetTaskRequest(id=str(uuid4()), params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, "📥 Query Task Response")


async def run_multi_turn_test(client: A2AClient) -> None:
    """
    Scenario 2: Multi-Turn Conversation
    Tests having a conversation where the agent might ask for clarification.
    """
    print("--- 📝 Multi-Turn Request ---")

    # --- Turn 1: Initial user query ---
    first_turn_payload = create_send_message_payload(text="How much is 100 in Kilogram?")
    request1 = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**first_turn_payload)
    )
    first_turn_response: SendMessageResponse = await client.send_message(request1)
    print_json_response(first_turn_response, "📥 Multi-Turn: First Turn Response")
    context_id: str | None = None

    # Check if the response is valid and get the Task details
    if isinstance(first_turn_response.root, SendMessageSuccessResponse) and isinstance(
            first_turn_response.root.result, Task
    ):
        task: Task = first_turn_response.root.result
        context_id = task.context_id  # Save context_id to link the next message
        # --- Turn 2: Reply (if input is required) ---
        # TaskState.input_required implies the agent is waiting for user info
        if task.status.state == TaskState.input_required and context_id:
            print("--- 📝 Multi-Turn: Second Turn (Input Required) ---")

            # Create payload LINKED to the previous task_id and context_id
            second_turn_payload = create_send_message_payload(
                "I mean 100 pounds", task.id, context_id
            )

            request2 = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**second_turn_payload)
            )

            # Send the clarification/follow-up message
            second_turn_response = await client.send_message(request2)
            print_json_response(
                second_turn_response, "Multi-Turn: Second Turn Response"
            )
        elif not context_id:
            print(
                "--- ⚠️ Warning: Could not get context ID from first turn response. ---"
            )
        else:
            print(
                "--- 🚀 First turn completed, no further input required for this test case. ---"
            )


async def main() -> None:
    """Entry point: Sets up connection and runs tests sequentially."""
    print(f"--- 🔄 Connecting to agent at {AGENT_URL}... ---")
    try:
        # Use httpx for async HTTP requests
        async with httpx.AsyncClient() as httpx_client:

            # 1. Resolve agent config (Card) from the URL
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=AGENT_URL,
            )
            agent_card = await resolver.get_agent_card()

            # 2. Instantiate the client using the resolved card
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=agent_card,
            )
            print("--- ✅ Connection successful. ---")
            # 3. Execute the tests
            await run_single_turn_test(client)
            # await run_multi_turn_test(client)
    except Exception as e:
        traceback.print_exc()
        print(f"--- ❌ An error occurred: {e} ---")
        print("Ensure the agent server is running.")


if __name__ == "__main__":
    import asyncio

    # Run the event loop
    asyncio.run(main())