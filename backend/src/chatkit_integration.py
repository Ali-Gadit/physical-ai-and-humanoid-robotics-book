from collections import defaultdict
from datetime import datetime
from typing import Any, AsyncIterator
from agents import Runner, Agent
from agents.extensions.models.litellm_model import LitellmModel
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import NotFoundError, Store
from chatkit.types import (
    Attachment, Page, ThreadItem, ThreadMetadata, 
    ThreadStreamEvent, UserMessageItem, AssistantMessageItem, AssistantMessageContent,
    ThreadItemAddedEvent, ThreadItemDoneEvent, ThreadItemUpdatedEvent
)
from chatkit.agents import AgentContext, stream_agent_response, ThreadItemConverter

# Import the agent components from chat_service
from src.services.chat_service import get_litellm_model, get_agent_instructions, search_textbook
# Import the new Postgres store
from src.store.postgres_store import PostgresChatKitStore

# --- Store Implementation (In-Memory for Demo) ---
# This class is now replaced by PostgresChatKitStore for T018,
# but kept here for historical context or if needed for other purposes.
class MyChatKitStore(Store[dict]):
    def __init__(self):
        self.threads: dict[str, ThreadMetadata] = {}
        self.items: dict[str, list[ThreadItem]] = defaultdict(list)

    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        if thread_id not in self.threads:
            # Auto-create thread if not found for simple demo
            return ThreadMetadata(id=thread_id, created_at=datetime.now())
        return self.threads[thread_id]

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        self.threads[thread.id] = thread

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: dict
    ) -> Page[ThreadMetadata]:
        threads = list(self.threads.values())
        return self._paginate(
            threads, after, limit, order, sort_key=lambda t: t.created_at, cursor_key=lambda t: t.id
        )

    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: dict
    ) -> Page[ThreadItem]:
        items = self.items.get(thread_id, [])
        return self._paginate(
            items, after, limit, order, sort_key=lambda i: i.created_at, cursor_key=lambda i: i.id
        )

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: dict
    ) -> None:
        self.items[thread_id].append(item)

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: dict
    ) -> None:
        items = self.items[thread_id]
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item
                return
        items.append(item)

    async def load_item(
        self, thread_id: str, item_id: str, context: dict
    ) -> ThreadItem:
        for item in self.items.get(thread_id, []):
            if item.id == item_id:
                return item
        raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        self.threads.pop(thread_id, None)
        self.items.pop(thread_id, None)

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: dict
    ) -> None:
        self.items[thread_id] = [
            item for item in self.items.get(thread_id, []) if item.id != item_id
        ]

    def _paginate(self, rows: list, after: str | None, limit: int, order: str, sort_key, cursor_key):
        sorted_rows = sorted(rows, key=sort_key, reverse=order == "desc")
        start = 0
        if after:
            for idx, row in enumerate(sorted_rows):
                if cursor_key(row) == after:
                    start = idx + 1
                    break
        data = sorted_rows[start : start + limit]
        has_more = start + limit < len(sorted_rows)
        next_after = cursor_key(data[-1]) if has_more and data else None
        return Page(data=data, has_more=has_more, after=next_after)

    async def save_attachment(
        self, attachment: Attachment, context: dict
    ) -> None:
        raise NotImplementedError()

    async def load_attachment(
        self, attachment_id: str, context: dict
    ) -> Attachment:
        raise NotImplementedError()

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        raise NotImplementedError()

# --- Server Implementation (from chatkit-agent-memory skill) ---
class ChatKitServerWithMemory(ChatKitServer[dict]):
    """ChatKit server with full conversation memory and LiteLLM ID fix"""

    def __init__(self, data_store: Store, model: LitellmModel, instructions: str):
        super().__init__(data_store)

        self.agent = Agent[AgentContext](
            name="RoboticsTutor", 
            instructions=instructions,
            model=model,
            tools=[search_textbook]
        )
        self.converter = ThreadItemConverter()

    async def respond(
        self,
        thread: ThreadMetadata,
        input: Any,
        context: dict
    ) -> AsyncIterator:
        """Generate response with full conversation context and unique IDs"""

        # Ensure thread exists in memory to avoid lookup errors
        # With a persistent store, we should attempt to load or create
        try:
            await self.store.load_thread(thread.id, context)
        except NotFoundError:
            await self.store.save_thread(thread, context)


        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )

        # Use 'desc' order to get latest items first, then reverse to 'asc' for agent context
        page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=100, 
            order="desc", 
            context=context
        )
        all_items = list(reversed(page.data))

        # The 'input' (UserMessageItem) is ALREADY added to the store by ChatKitServer.process
        # before respond() is called. 
        # So 'all_items' loaded from store SHOULD contain the user input.
        print(f"[Server] Loaded {len(all_items)} items from store for thread {thread.id}")

        agent_input = await self.converter.to_agent_input(all_items) if all_items else []

        print(f"[Server] Converted to {len(agent_input)} agent input items")

        result = Runner.run_streamed(
            self.agent,
            agent_input,
            context=agent_context,
        )

        id_mapping: dict[str, str] = {}

        async for event in stream_agent_response(agent_context, result):
            if event.type == "thread.item.added":
                if isinstance(event.item, AssistantMessageItem):
                    old_id = event.item.id
                    if old_id not in id_mapping:
                        new_id = self.store.generate_item_id("message", thread, context)
                        id_mapping[old_id] = new_id
                        print(f"[Server] Mapping ID {old_id} -> {new_id}")
                    event.item.id = id_mapping[old_id]

            elif event.type == "thread.item.done":
                if isinstance(event.item, AssistantMessageItem):
                    old_id = event.item.id
                    if old_id in id_mapping:
                        event.item.id = id_mapping[old_id]

            elif event.type == "thread.item.updated":
                if event.item_id in id_mapping:
                    event.item_id = id_mapping[event.item_id]

            yield event

# Instantiate the server using the refactored chat_service components
chatkit_server = ChatKitServerWithMemory(
    data_store=PostgresChatKitStore(), # Use the Postgres store here
    model=get_litellm_model(),
    instructions=get_agent_instructions()
)