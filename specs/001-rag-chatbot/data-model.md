# Data Model: Integrated RAG Chatbot

## Overview
Data models for the RAG chatbot system, covering conversation management, content storage, and user interactions.

## Entity Models

### 1. Conversation
**Purpose**: Track chat sessions between users and the AI assistant
- `id` (UUID): Unique conversation identifier
- `user_id` (UUID, optional): User identifier if authenticated
- `created_at` (DateTime): Conversation start time
- `updated_at` (DateTime): Last interaction time
- `title` (String): Auto-generated conversation title
- `metadata` (JSON): Additional conversation properties

### 2. Message
**Purpose**: Store individual messages within conversations
- `id` (UUID): Unique message identifier
- `conversation_id` (UUID): Reference to parent conversation
- `role` (String): Message role (user, assistant, system)
- `content` (Text): Message content
- `timestamp` (DateTime): When message was created
- `source_chunks` (JSON): Vector IDs of source content used for response
- `feedback` (Integer, optional): User feedback (1-5 scale)

### 3. ContentChunk
**Purpose**: Store book content in vectorized chunks for RAG retrieval
- `id` (UUID): Unique chunk identifier
- `document_id` (String): Reference to source document/chapter
- `chunk_index` (Integer): Position within document
- `content` (Text): Chunk text content
- `embedding` (Vector): Gemini-generated embedding vector
- `metadata` (JSON): Additional properties (page, section, etc.)

### 4. Document
**Purpose**: Track book documents/chapters for content management
- `id` (String): Document identifier (e.g., chapter slug)
- `title` (String): Document title
- `url_path` (String): Path in the book structure
- `version` (String): Content version
- `created_at` (DateTime): When document was first processed
- `updated_at` (DateTime): When document was last processed
- `word_count` (Integer): Number of words in document
- `status` (String): Processing status (pending, processed, failed)

### 5. UserSession
**Purpose**: Track anonymous user sessions for personalization
- `id` (UUID): Session identifier
- `start_time` (DateTime): Session start
- `end_time` (DateTime, optional): Session end
- `ip_address` (String, optional): User IP for analytics
- `user_agent` (String, optional): Browser information

### 6. QueryLog
**Purpose**: Log user queries for analytics and improvement
- `id` (UUID): Log entry identifier
- `user_id` (UUID, optional): User identifier
- `session_id` (UUID): Session identifier
- `query` (Text): Original user query
- `response` (Text): AI response
- `retrieved_chunks` (JSON): IDs of content chunks used
- `timestamp` (DateTime): When query was made
- `response_time` (Float): Time taken to generate response
- `satisfaction_score` (Integer, optional): User satisfaction rating

## Relationships

### Conversation ↔ Message
- One-to-Many: A conversation contains multiple messages
- Cascade delete: Messages deleted when conversation is deleted

### Document ↔ ContentChunk
- One-to-Many: A document contains multiple content chunks
- Cascade delete: Chunks deleted when document is deleted

## Validation Rules

### Conversation Model
- `title` must be 1-100 characters
- `created_at` cannot be in the future
- `updated_at` must be >= `created_at`

### Message Model
- `role` must be one of: "user", "assistant", "system"
- `content` must be 1-10000 characters
- `conversation_id` must reference an existing conversation

### ContentChunk Model
- `content` must be 50-2000 characters (for optimal RAG performance)
- `embedding` vector must have correct dimensions for the embedding model
- `chunk_index` must be non-negative

### Document Model
- `id` must be unique
- `url_path` must follow Docusaurus URL conventions
- `status` must be one of: "pending", "processed", "failed"

## State Transitions

### Document Status
- `pending` → `processed`: When content is successfully vectorized
- `pending` → `failed`: When content processing fails
- `processed` → `pending`: When content is updated and needs reprocessing

## Indexes

### Performance Optimizations
- Index on `Message.conversation_id` for conversation queries
- Index on `ContentChunk.document_id` for document retrieval
- Composite index on `QueryLog.timestamp, QueryLog.user_id` for analytics
- Vector index on `ContentChunk.embedding` for similarity search