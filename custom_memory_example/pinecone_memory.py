

"""Pinecone-based semantic memory service for ADK.

This service stores conversational Q&A pairs in Pinecone with embeddings,
enabling semantic search to retrieve answers for similar questions.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from google.adk.memory import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions import Session
from google.genai import Client
from pinecone import Pinecone, ServerlessSpec

from typing_extensions import override

from . import _utils


if TYPE_CHECKING:
    from google.genai.types import Content


logger = logging.getLogger(__name__)


class PineconeSemanticMemoryService(BaseMemoryService):
    """Pinecone-based semantic memory service.

    Stores user inputs as embeddings in Pinecone with associated answers
    and metadata. Enables semantic search to retrieve relevant Q&A pairs
    for similar questions.

    Architecture:
        - Extracts Q&A pairs from session events
        - Generates embeddings for user inputs using Google GenAI
        - Stores embeddings in Pinecone with metadata (answer, timestamp, etc)
        - Performs similarity search to find relevant memories

    Pinecone Schema:
        - id: {session_id}_{event_idx}
        - values: embedding vector (768 dimensions)
        - metadata:
            - app_name: application name
            - user_id: user identifier
            - session_id: session identifier
            - input_text: original user question
            - answer_text: assistant's answer
            - author: event author
            - timestamp: ISO 8601 timestamp
            - custom_metadata: additional metadata dict

    Environment Variables Required:
        - PINECONE_API_KEY: Pinecone API key
        - PINECONE_INDEX_NAME: Name of the Pinecone index (default: adk-semantic-memory)
        - GOOGLE_API_KEY: Google GenAI API key for embeddings
    """

    def __init__(
        self,
        *,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_dimension: int = 3072,
        top_k: int = 5,
        create_index_if_not_exists: bool = True,
    ):
        """Initialize the Pinecone semantic memory service.

        Args:
            pinecone_api_key: Pinecone API key. If not provided, reads from
                PINECONE_API_KEY environment variable.
            index_name: Name of the Pinecone index. If not provided, reads from
                PINECONE_INDEX_NAME environment variable or uses default.
            embedding_model: Google GenAI embedding model to use.
            embedding_dimension: Dimension of the embedding vectors.
            top_k: Number of top results to return in semantic search.
            create_index_if_not_exists: Whether to create the index if it doesn't exist.

        Raises:
            ValueError: If required API keys are not provided.
        """
        # Get API keys
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError(
                "Pinecone API key must be provided or set in PINECONE_API_KEY "
                "environment variable"
            )

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "Google GenAI API key must be set in GOOGLE_API_KEY "
                "environment variable"
            )

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "adk-semantic-memory"
        )
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.top_k = top_k

        # Initialize Google GenAI client for embeddings
        self.genai_client = Client(api_key=google_api_key)

        # Create or connect to index
        if create_index_if_not_exists:
            self._ensure_index_exists()

        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
            logger.info(f"Index {self.index_name} created successfully")

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for the given text using Google GenAI.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        response = await self.genai_client.aio.models.embed_content(
            model=self.embedding_model,
            contents=[text],  # Note: contents expects a list
        )
        return response.embeddings[0].values

    def _extract_qa_pairs(
        self, session: Session
    ) -> list[tuple[Content, Content, int]]:
        """Extract question-answer pairs from session events.

        Pairs consecutive user and assistant/model events as Q&A pairs.

        Args:
            session: The session to extract Q&A pairs from.

        Returns:
            List of tuples: (user_content, assistant_content, event_index)
        """
        qa_pairs = []
        events = [
            (i, event)
            for i, event in enumerate(session.events)
            if event.content and event.content.parts
        ]

        i = 0
        while i < len(events) - 1:
            idx_i, event_i = events[i]
            idx_j, event_j = events[i + 1]

            # Check if current is user and next is assistant/model
            if (
                event_i.content.role == "user"
                and event_j.content.role in ["model", "assistant"]
            ):
                qa_pairs.append((event_i.content, event_j.content, idx_i))
                i += 2  # Skip both events
            else:
                i += 1  # Move to next event

        print("QA pairs:", qa_pairs)
        return qa_pairs

    @override
    async def add_session_to_memory(self, session: Session):
        """Add a session to the Pinecone memory.

        Extracts Q&A pairs from the session, generates embeddings for questions,
        and stores them in Pinecone with answers in metadata.

        Args:
            session: The session to add to memory.
        """
        qa_pairs = self._extract_qa_pairs(session)

        if not qa_pairs:
            logger.info(f"No Q&A pairs found in session {session.id}")
            return

        vectors_to_upsert = []

        for user_content, assistant_content, event_idx in qa_pairs:
            # Extract text from content
            input_text = " ".join(
                [part.text for part in user_content.parts if part.text]
            )
            answer_text = " ".join(
                [part.text for part in assistant_content.parts if part.text]
            )

            if not input_text or not answer_text:
                continue

            # Generate embedding for the user input
            embedding = await self._generate_embedding(input_text)

            # Create unique ID
            vector_id = f"{session.id}_{event_idx}"

            # Prepare metadata
            metadata = {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "input_text": input_text,
                "answer_text": answer_text,
                "author": session.events[event_idx].author or "user",
                "timestamp": _utils.format_timestamp(
                    session.events[event_idx].timestamp
                ),
            }

            # Add custom metadata if available
            if hasattr(session.events[event_idx], "custom_metadata"):
                metadata["custom_metadata"] = str(
                    session.events[event_idx].custom_metadata
                )

            vectors_to_upsert.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        # Upsert vectors to Pinecone
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            logger.info(
                f"Added {len(vectors_to_upsert)} Q&A pairs from session "
                f"{session.id} to Pinecone"
            )

    @override
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search for relevant memories using semantic similarity.

        Generates an embedding for the query and performs similarity search
        in Pinecone to find relevant Q&A pairs.

        Args:
            app_name: The name of the application.
            user_id: The id of the user.
            query: The query to search for.

        Returns:
            A SearchMemoryResponse containing matching memories.
        """
        # Generate embedding for the query
        query_embedding = await self._generate_embedding(query)

        # Search Pinecone with filters
        search_results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True,
            filter={
                "app_name": {"$eq": app_name},
                "user_id": {"$eq": user_id},
            },
        )

        # Convert results to MemoryEntry objects
        memories = []
        for match in search_results.matches:
            metadata = match.metadata

            # Create a Content object with the answer text
            from google.genai.types import Content, Part

            answer_content = Content(
                role="model",
                parts=[Part.from_text(text=metadata.get("answer_text", ""))],
            )

            memory_entry = MemoryEntry(
                content=answer_content,
                id=match.id,
                author=metadata.get("author"),
                timestamp=metadata.get("timestamp"),
                custom_metadata={
                    "input_text": metadata.get("input_text"),
                    "similarity_score": match.score,
                    "session_id": metadata.get("session_id"),
                },
            )
            memories.append(memory_entry)

        logger.info(
            f"Found {len(memories)} relevant memories for query: '{query}'"
        )
        return SearchMemoryResponse(memories=memories)
