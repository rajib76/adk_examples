from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import ToolContext

MODEL = "gemini-2.0-flash"
load_dotenv()


def set_state_color(color: str, tool_context: ToolContext) -> dict:
    """Set the session-level 'color' state."""
    tool_context.state["color"] = color
    return {"status": "ok", "color": color}


def get_state_color(tool_context: ToolContext) -> dict:
    """Read the session-level 'color' state."""
    return {"color": tool_context.state.get("color")}


root_agent = Agent(
    name="color_state_agent",
    model=MODEL,
    description="Manages a session state key named 'color'.",
    instruction=(
        "You are a stateful agent.\n"
        "The session state contains a key called 'color'.\n"
        "- If the user asks to set/update/change the color, call set_state_color(color=...).\n"
        "- If the user asks for the current color, call get_state_color().\n"
        "Reply concisely with the value."
    ),
    tools=[set_state_color, get_state_color],
)
