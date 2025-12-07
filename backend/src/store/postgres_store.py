import json
import uuid
from datetime import datetime
from typing import AsyncIterator, List, TypeVar, Generic, Any, Callable

from chatkit.store import Store, NotFoundError
from chatkit.types import (
    ThreadMetadata, ThreadItem, Page, AssistantMessageItem, UserMessageItem,
    Attachment, ThreadItemAddedEvent, ThreadItemDoneEvent, ThreadItemUpdatedEvent
)
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text, asc, desc
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
from sqlalchemy.exc import NoResultFound

from src.db.postgres import get_db_connection
from src.models.chat import Base, Thread, ThreadItem as DBThreadItem

TContext = TypeVar("TContext")

# Database setup (to be used within the store)
engine = None
SessionLocal = None

def init_db_engine():
    global engine, SessionLocal
    if engine is None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        db_url = os.getenv("NEON_DATABASE_URL")
        if not db_url:
            raise ValueError("NEON_DATABASE_URL not found in environment variables.")
        # Ensure it's a valid SQLAlchemy URL (e.g., replace psql with postgresql)
        if db_url.startswith("psql"):
            db_url = db_url.replace("psql", "postgresql", 1)
        engine = create_engine(db_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        print("Database engine and tables initialized.")

def get_db_session() -> Session:
    if SessionLocal is None:
        init_db_engine()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

class PostgresChatKitStore(Store[TContext]):
    def __init__(self):
        # Initialize DB engine and create tables once
        init_db_engine()

    def _convert_db_to_chatkit_thread(self, db_thread: Thread) -> ThreadMetadata:
        return ThreadMetadata(
            id=db_thread.id,
            created_at=db_thread.created_at,
            # Add other fields as needed
        )

    def _convert_chatkit_to_db_thread(self, chatkit_thread: ThreadMetadata) -> Thread:
        return Thread(
            id=chatkit_thread.id,
            created_at=chatkit_thread.created_at,
        )

    def _convert_db_to_chatkit_item(self, db_item: DBThreadItem) -> ThreadItem:
        # Deserialize the payload_json to the correct ChatKit ThreadItem type
        payload = json.loads(db_item.payload_json)
        # Reconstruct the specific ThreadItem subclass
        if db_item.item_type == "UserMessageItem":
            return UserMessageItem.model_validate(payload)
        elif db_item.item_type == "AssistantMessageItem":
            return AssistantMessageItem.model_validate(payload)
        # Add other types as needed
        else:
            # Fallback or raise error for unknown types
            return ThreadItem.model_validate(payload)

    def _convert_chatkit_to_db_item(self, chatkit_item: ThreadItem) -> DBThreadItem:
        return DBThreadItem(
            id=chatkit_item.id,
            thread_id=chatkit_item.thread_id,
            item_type=chatkit_item.__class__.__name__, # Store class name to reconstruct
            content_text=str(chatkit_item.content) if hasattr(chatkit_item, 'content') else None,
            payload_json=chatkit_item.model_dump_json(), # Store full JSON representation
            created_at=chatkit_item.created_at,
        )

    def generate_thread_id(self, context: TContext) -> str:
        return str(uuid.uuid4())

    def generate_item_id(self, item_type: str, thread: ThreadMetadata, context: TContext) -> str:
        return str(uuid.uuid4())

    async def load_thread(self, thread_id: str, context: TContext) -> ThreadMetadata:
        with next(get_db_session()) as session:
            try:
                db_thread = session.query(Thread).filter_by(id=thread_id).one()
                return self._convert_db_to_chatkit_thread(db_thread)
            except NoResultFound:
                # Automatically create the thread if it doesn't exist
                # This mimics the behavior of an in-memory store that implicitly "creates" threads
                # when they are first accessed or saved.
                # This resolves the "NotFoundError" when ChatKit tries to load a new thread ID.
                new_thread = Thread(id=thread_id, created_at=datetime.now())
                session.add(new_thread)
                session.commit()
                print(f"[Store] Thread {thread_id} not found, created new one.")
                return self._convert_db_to_chatkit_thread(new_thread)

    async def save_thread(self, thread: ThreadMetadata, context: TContext) -> None:
        with next(get_db_session()) as session:
            db_thread = session.query(Thread).filter_by(id=thread.id).first()
            if db_thread:
                db_thread.created_at = thread.created_at # Update existing
            else:
                db_thread = self._convert_chatkit_to_db_thread(thread)
                session.add(db_thread)
            session.commit()

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: TContext
    ) -> Page[ThreadMetadata]:
        with next(get_db_session()) as session:
            query = session.query(Thread)
            if order == "desc":
                query = query.order_by(desc(Thread.created_at))
            else:
                query = query.order_by(asc(Thread.created_at))

            if after:
                last_thread = session.query(Thread).filter_by(id=after).first()
                if last_thread:
                    if order == "desc":
                        query = query.filter(Thread.created_at < last_thread.created_at)
                    else:
                        query = query.filter(Thread.created_at > last_thread.created_at)

            db_threads = query.limit(limit + 1).all()
            has_more = len(db_threads) > limit
            data = [self._convert_db_to_chatkit_thread(t) for t in db_threads[:limit]]
            next_after = data[-1].id if has_more and data else None
            return Page(data=data, has_more=has_more, after=next_after)

    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: TContext
    ) -> Page[ThreadItem]:
        with next(get_db_session()) as session:
            query = session.query(DBThreadItem).filter_by(thread_id=thread_id)
            if order == "desc":
                query = query.order_by(desc(DBThreadItem.created_at))
            else:
                query = query.order_by(asc(DBThreadItem.created_at))
            
            if after:
                last_item = session.query(DBThreadItem).filter_by(id=after).first()
                if last_item:
                    if order == "desc":
                        query = query.filter(DBThreadItem.created_at < last_item.created_at)
                    else:
                        query = query.filter(DBThreadItem.created_at > last_item.created_at)

            db_items = query.limit(limit + 1).all()
            has_more = len(db_items) > limit
            data = [self._convert_db_to_chatkit_item(item) for item in db_items[:limit]]
            next_after = data[-1].id if has_more and data else None
            return Page(data=data, has_more=has_more, after=next_after)


    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: TContext
    ) -> None:
        with next(get_db_session()) as session:
            # Ensure thread exists before adding item
            try:
                session.query(Thread).filter_by(id=thread_id).one()
            except NoResultFound:
                 # Auto-create thread if missing (safety net)
                 new_thread = Thread(id=thread_id, created_at=datetime.now())
                 session.add(new_thread)
                 session.commit()
                 print(f"[Store] Auto-created missing thread {thread_id} before adding item.")

            db_item = self._convert_chatkit_to_db_item(item)
            session.add(db_item)
            session.commit()

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: TContext
    ) -> None:
        with next(get_db_session()) as session:
            db_item = session.query(DBThreadItem).filter_by(id=item.id).first()
            if db_item:
                db_item.thread_id = item.thread_id
                db_item.item_type = item.__class__.__name__
                db_item.content_text = str(item.content) if hasattr(item, 'content') else None
                db_item.payload_json = item.model_dump_json()
                db_item.created_at = item.created_at
            else:
                db_item = self._convert_chatkit_to_db_item(item)
                session.add(db_item)
            session.commit()

    async def load_item(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ThreadItem:
        with next(get_db_session()) as session:
            try:
                db_item = session.query(DBThreadItem).filter_by(id=item_id, thread_id=thread_id).one()
                return self._convert_db_to_chatkit_item(db_item)
            except NoResultFound:
                raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")

    async def delete_thread(self, thread_id: str, context: TContext) -> None:
        with next(get_db_session()) as session:
            session.query(DBThreadItem).filter_by(thread_id=thread_id).delete()
            session.query(Thread).filter_by(id=thread_id).delete()
            session.commit()

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: TContext
    ) -> None:
        with next(get_db_session()) as session:
            session.query(DBThreadItem).filter_by(id=item_id, thread_id=thread_id).delete()
            session.commit()

    async def save_attachment(self, attachment: Attachment, context: TContext) -> None:
        raise NotImplementedError("Attachment storage not implemented for Postgres store.")

    async def load_attachment(self, attachment_id: str, context: TContext) -> Attachment:
        raise NotImplementedError("Attachment loading not implemented for Postgres store.")

    async def delete_attachment(self, attachment_id: str, context: TContext) -> None:
        raise NotImplementedError("Attachment deletion not implemented for Postgres store.")
