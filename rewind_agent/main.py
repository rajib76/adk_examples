import asyncio


from google.adk.runners import InMemoryRunner
from google.genai import types

from rewind_agent.agent import root_agent

APP_NAME = "rewind_demo_app"
USER_ID = "rajib"  # change if you want


async def call_agent_async(runner: InMemoryRunner, user_id: str, session_id: str, query: str):
    """
    Wrapper that sends one user message, prints the final response, and
    returns the full list of events (so you can grab invocation_id).
    """
    print(f"\n>>> User: {query}")
    content = types.Content(role="user", parts=[types.Part(text=query)])

    events = []
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        events.append(event)

        if event.is_final_response():
            text = ""
            if event.content and event.content.parts:
                text = event.content.parts[0].text or ""
            print(f"<<< Agent: {text}".rstrip())
            break

    return events


async def main():
    # Create runner
    runner = InMemoryRunner(
        agent=root_agent,
        app_name=APP_NAME,
    )

    # Create a session
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )

    # Set color to red"
    events_list= await call_agent_async(
        runner, USER_ID, session.id, "set state color to red"
    )
    rewind_invocation_id_0 = events_list[0].invocation_id
    print(f"Rewind invocation id: {rewind_invocation_id_0}")

    # Set color to orange"
    events_list= await call_agent_async(
        runner, USER_ID, session.id, "set state color to orange"
    )
    rewind_invocation_id_1 = events_list[0].invocation_id
    print(f"Rewind invocation id: {rewind_invocation_id_1}")

    # Set color to blue
    events_list = await call_agent_async(
        runner, USER_ID, session.id, "update state color to blue"
    )


    rewind_invocation_id_2 = events_list[1].invocation_id
    print(f"Rewind invocation id: {rewind_invocation_id_2}")

    # rewind invocations (state color should go back to: red)
    await runner.rewind_async(
        user_id=USER_ID,
        session_id=session.id,
        rewind_before_invocation_id=rewind_invocation_id_2,
    )

    # sanity check after rewind
    await call_agent_async(runner, USER_ID, session.id, "what is the current state color?")


if __name__ == "__main__":
    asyncio.run(main())
