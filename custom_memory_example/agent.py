"""
Pinecone Semantic Memory Demo
==============================
This program demonstrates storage and retrieval of semantic memory using Pinecone.

Setup:
1. Install: pip install -r requirements.txt
2. Set environment variables:
   - PINECONE_API_KEY=your_pinecone_api_key
   - GOOGLE_API_KEY=your_google_api_key
3. Run: python demo_pinecone_memory.py
"""

import asyncio

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_memory_tool import load_memory_tool
from google.genai import types

from custom_memory_example.pinecone_memory import PineconeSemanticMemoryService

# Load environment variables from .env file
load_dotenv()

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# Create an agent with memory tools
demo_agent = Agent(
    model="gemini-2.0-flash-001",
    name="memory_demo_agent",
    description="Agent demonstrating Pinecone semantic memory",
    instruction="""\
You are a helpful assistant with access to semantic memory.

When asked questions, use the load_memory tool to search for relevant information
from past conversations. The memory is semantic, so you can find answers even if
the question is phrased differently than the original.
""",
    tools=[load_memory_tool],
)


async def send_message(runner, session_id: str, user_id: str, message: str):
    """Send a message and collect the response."""
    print(f"\n👤 USER: {message}")

    content = types.Content(
        role="user", parts=[types.Part.from_text(text=message)]
    )

    responses = []
    async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if part.text:
                print(f"🤖 AGENT: {part.text}")
                responses.append(part.text)
            elif part.function_call:
                print(f"🔧 TOOL CALL: {part.function_call.name}")
            elif part.function_response:
                print(f"📦 TOOL RESPONSE: (semantic search completed)")

    return responses


async def main():
    """Run the Pinecone semantic memory demonstration."""

    print_section("PINECONE SEMANTIC MEMORY DEMONSTRATION")
    print("\nThis demo shows how semantic memory stores and retrieves information")
    print("based on meaning, not just keywords.\n")

    app_name = "pinecone_demo"
    user_id = "demo_user"

    # Initialize Pinecone memory service
    print_section("STEP 1: Initialize Pinecone Semantic Memory Service")
    print("\n⚙️  Creating PineconeSemanticMemoryService...")
    print("   - Connecting to Pinecone")
    print("   - Creating/verifying index: 'adk-demo-memory'")
    print("   - Initializing embedding model: gemini-embedding-001")

    memory_service = PineconeSemanticMemoryService(
        index_name="adk-demo-memory",
        top_k=3,  # Return top 3 most similar results
    )
    session_service = InMemorySessionService()
    print("✅ Memory service initialized successfully!")

    # Create runner with memory service
    runner = Runner(
        app_name=app_name,
        agent=demo_agent,
        session_service=session_service,
        memory_service=memory_service,
    )

    # ========================================================================
    # PHASE 1: STORAGE - Store memories
    # ========================================================================
    print_section("STEP 2: STORAGE - Storing Memories in Pinecone")

    print("\n📝 Creating a conversation with factual Q&A pairs...")
    print("   (Each Q&A pair will be embedded and stored in Pinecone)\n")

    # Create a session for storing memories
    storage_session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    print("💬 Conversation 1: Personal Information")
    await send_message(runner, storage_session.id, user_id, "Hello!")
    await send_message(runner, storage_session.id, user_id, "My name is John Smith")
    await send_message(runner, storage_session.id, user_id, "I work as a data scientist")
    await send_message(runner, storage_session.id, user_id, "My favorite programming language is Python")
    await send_message(runner, storage_session.id, user_id, "I love playing basketball on weekends")

    print("\n💾 STORING SESSION TO PINECONE...")
    print("   Processing:")
    print("   1. Extracting Q&A pairs from conversation")
    print("   2. Generating embeddings for each question")
    print("   3. Storing vectors with answers in metadata")
    print("   4. Uploading to Pinecone index")

    # Get the updated session with all events
    storage_session = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=storage_session.id
    )

    # Add session to Pinecone memory
    await memory_service.add_session_to_memory(storage_session)

    print("\n✅ STORAGE COMPLETE!")
    print("   ✓ Q&A pairs extracted and embedded")
    print("   ✓ Vectors stored in Pinecone")
    print("   ✓ Answers stored in metadata")
    print("   ✓ Ready for semantic search!")

    # ========================================================================
    # PHASE 2: RETRIEVAL - Retrieve memories with semantic search
    # ========================================================================
    print_section("STEP 3: RETRIEVAL - Semantic Search in Action")

    print("\n🔍 Now we'll ask SIMILAR questions (different wording)")
    print("   The agent will use semantic search to find relevant answers\n")

    # Create a NEW session for retrieval
    retrieval_session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    print("─" * 80)
    print("TEST 1: Query about name (using different words)")
    print("   Original: 'My name is John Smith'")
    print("   Query:    'What's my name?'")
    await send_message(runner, retrieval_session.id, user_id, "What's my name?")

    print("\n" + "─" * 80)
    print("TEST 2: Query about job (different phrasing)")
    print("   Original: 'I work as a data scientist'")
    print("   Query:    'What is my occupation?'")
    await send_message(runner, retrieval_session.id, user_id, "What is my occupation?")

    print("\n" + "─" * 80)
    print("TEST 3: Query about hobby (semantic similarity)")
    print("   Original: 'I love playing basketball on weekends'")
    print("   Query:    'Which sport do I enjoy?'")
    await send_message(runner, retrieval_session.id, user_id, "Which sport do I enjoy?")

    print("\n" + "─" * 80)
    print("TEST 4: Query about programming (different words)")
    print("   Original: 'My favorite programming language is Python'")
    print("   Query:    'What coding language do I prefer?'")
    await send_message(runner, retrieval_session.id, user_id, "What coding language do I prefer?")

    # ========================================================================
    # PHASE 3: Add more memories and test again
    # ========================================================================
    print_section("STEP 4: Adding More Memories")

    print("\n📝 Creating another conversation with new information...\n")

    storage_session_2 = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    print("💬 Conversation 2: Additional Information")
    await send_message(runner, storage_session_2.id, user_id, "I graduated from Stanford University")
    await send_message(runner, storage_session_2.id, user_id, "My favorite food is sushi")
    await send_message(runner, storage_session_2.id, user_id, "I live in San Francisco")

    print("\n💾 STORING ADDITIONAL MEMORIES TO PINECONE...")

    storage_session_2 = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=storage_session_2.id
    )

    await memory_service.add_session_to_memory(storage_session_2)

    print("✅ Additional memories stored!")

    # ========================================================================
    # Test retrieval of new memories
    # ========================================================================
    print_section("STEP 5: Retrieving New Memories")

    print("\n🔍 Testing retrieval of newly added information...\n")

    retrieval_session_2 = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    print("─" * 80)
    print("TEST 5: Query about education")
    print("   Original: 'I graduated from Stanford University'")
    print("   Query:    'Where did I study?'")
    await send_message(runner,retrieval_session_2.id, user_id, "Where did I study?")

    print("\n" + "─" * 80)
    print("TEST 6: Query about location")
    print("   Original: 'I live in San Francisco'")
    print("   Query:    'What city do I reside in?'")
    await send_message(runner,retrieval_session_2.id, user_id, "What city do I reside in?")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("DEMONSTRATION COMPLETE")

    print("""
✨ KEY TAKEAWAYS:

1. STORAGE:
   - Q&A pairs are automatically extracted from conversations
   - Questions are converted to embeddings (3072-dim vectors)
   - Answers are stored in Pinecone metadata
   - Everything is indexed for fast semantic search

2. RETRIEVAL:
   - Queries are converted to embeddings
   - Pinecone finds semantically similar questions
   - Answers are retrieved from metadata
   - Works even with different wording!

3. SEMANTIC UNDERSTANDING:
   - "What's my name?" ≈ "My name is..."
   - "What is my occupation?" ≈ "I work as..."
   - "Which sport do I enjoy?" ≈ "I love playing..."

This is the power of semantic memory! 🚀
""")



if __name__ == "__main__":
    print("\n🚀 Starting Pinecone Semantic Memory Demo...\n")
    asyncio.run(main())
