from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime

# Base for declarative models
Base = declarative_base()

class Thread(Base):
    __tablename__ = 'chat_threads'

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    # Add any other thread-level metadata here, e.g., user_id
    
    # Relationship to thread items
    items = relationship("ThreadItem", back_populates="thread", order_by="ThreadItem.created_at")

    def __repr__(self):
        return f"<Thread(id='{self.id}', created_at='{self.created_at}')>"

class ThreadItem(Base):
    __tablename__ = 'chat_thread_items'

    id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey('chat_threads.id'))
    item_type = Column(String, nullable=False) # e.g., 'user_message', 'assistant_message'
    content_text = Column(Text) # Main text content
    payload_json = Column(Text) # Store full JSON payload of ChatKit ThreadItem
    created_at = Column(DateTime, default=datetime.now)

    # Relationship to thread
    thread = relationship("Thread", back_populates="items")

    def __repr__(self):
        return f"<ThreadItem(id='{self.id}', type='{self.item_type}', thread_id='{self.thread_id}')>"

# This engine and SessionLocal is for direct use, if needed for migrations or manual ops.
# For FastAPI dependency, it will be handled by get_db_connection.
# engine = create_engine(os.getenv("NEON_DATABASE_URL")) # Need to get from env
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create tables (for initial setup, can be replaced by Alembic)
def create_db_tables(engine):
    Base.metadata.create_all(bind=engine)
