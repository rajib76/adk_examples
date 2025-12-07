from a2a.types import AgentCard, AgentCapabilities
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.llm_agent import Agent

load_dotenv()

my_agent_capability = AgentCapabilities(
    extensions= None,
    push_notifications= False,
    state_transition_history=True,
    streaming=True
)

my_agent_card = AgentCard(name="qa_agent",
                          url="http://localhost:8001",
                          description="A helpful assistant for user questions. Answer user questions to the best of my knowledge",
                          version="0.0.1",
                          capabilities=my_agent_capability,
                          skills=[],
                          default_input_modes=["text/plain"],
                          default_output_modes=["text/plain"],
                          supports_authenticated_extended_card=False,
)

root_agent = Agent(
    model='gemini-2.0-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)

a2a_app = to_a2a(root_agent, port=8001,agent_card=my_agent_card)
# a2a_app = to_a2a(root_agent, port=8001)