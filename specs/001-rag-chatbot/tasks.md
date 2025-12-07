# Tasks: Integrated RAG Chatbot for Physical AI and Humanoid Robotics Textbook

**Feature Branch**: `001-rag-chatbot`
**Status**: Pending
**Spec**: [specs/001-rag-chatbot/spec.md](specs/001-rag-chatbot/spec.md)
**Plan**: [specs/001-rag-chatbot/plan.md](specs/001-rag-chatbot/plan.md)

## Phase 1: Setup
*Goal: Initialize project structure and configure environment for Backend and Frontend components.*

- [x] T001 Initialize Python FastAPI project structure in `backend/` (src, tests, requirements.txt)
- [x] T002 [P] Initialize React/library project in `frontend/` for Chatbot widget
- [x] T003 Set up environment variables `.env` for Gemini API, Qdrant URL/Key, Neon DB URL in `backend/`
- [x] T004 [P] Fetch and analyze OpenAI Agents SDK and ChatKit documentation via Context7 to guide implementation (save notes to `specs/001-rag-chatbot/research.md`)
- [x] T005 [P] Implement Neon Postgres database connection logic in `backend/src/db/postgres.py`
- [x] T006 [P] Implement Qdrant vector store connection logic in `backend/src/rag/vector_store.py`

## Phase 2: Foundational (Blocking)
*Goal: Ingest book content and establish core backend services required for RAG.*

- [x] T007 Implement content ingestion service in `backend/src/services/content_service.py` to read markdown from `docusaurus-book/docs`
- [x] T008 Implement text chunking and preprocessing logic in `backend/src/rag/processing.py`
- [x] T009 Implement Gemini embedding generation using Google GenAI SDK in `backend/src/rag/embedding.py`
- [x] T010 Create script `backend/scripts/ingest_content.py` and run it to populate Qdrant with book content

## Phase 3: Interactive Book Q&A (User Story 1)
*Goal: Enable users to ask questions about book content and receive accurate answers.*
*Priority: P1*

- [x] T011 [US1] Implement RAG retrieval logic (search Qdrant with Gemini embeddings) in `backend/src/rag/retrieval.py`
- [x] T012 [US1] Implement basic Chat Agent using OpenAI Agents SDK (configured with Gemini models) in `backend/src/services/chat_service.py`
- [x] T013 [US1] Create Chat API endpoints (`/chatkit`) using `ChatKitServer` in `backend/src/api/main.py`
- [x] T014 [US1] Create Chatbot UI component using OpenAI ChatKit in `docusaurus-book/src/components/Chatbot/Chatbot.jsx`
- [x] T015 [US1] Create ChatButton component for toggling the chat window in `docusaurus-book/src/components/Chatbot/ChatButton.jsx`
- [x] T016 [US1] Integrate Chatbot widget into Docusaurus theme in `docusaurus-book/src/theme/Layout/index.js`
- [x] T017 [US1] Verify end-to-end Q&A flow with manual testing

## Phase 4: Context-Aware Question Answering (User Story 2)
*Goal: Enable multi-turn conversations with context retention.*
*Priority: P2*

- [x] T018 [US2] Implement conversation session management in `backend/src/models/chat.py` (store in Neon DB)
- [x] T019 [US2] Update `chat_service.py` to retrieve and use conversation history for context-aware responses
- [x] T020 [US2] Update Frontend API client in `frontend/src/services/api.js` to handle and persist Session IDs
- [x] T021 [US2] Add session reset/clear functionality to Chatbot UI in `frontend/src/components/Chatbot.jsx`

## Phase 5: Text Selection and Focused Q&A (User Story 3) 
*Goal: Allow users to select text and ask questions specifically about that selection.*
*Priority: P3*

- [ ] T022 [US3] Create TextSelection handler component in `docusaurus-book/src/components/Chatbot/TextSelectionHandler.jsx` to display a floating "Ask Assistant" tooltip on selection.
- [ ] T023 [US3] Update Chatbot UI and `ChatButton.jsx` to trigger chatbot with selected text from floating tooltip.
- [ ] T024 [US3] Update Chat API in `backend/src/api/chat.py` to accept `selected_text` in request body
- [ ] T025 [US3] Update Agent logic in `backend/src/services/chat_service.py` to prioritize or restrict context to `selected_text` when provided

## Phase 6: Polish & Cross-Cutting
*Goal: Refine UI/UX, ensure robustness, and prepare for deployment.*

- [x] T026 Style Chatbot UI to match Textbook theme (colors, fonts) in `frontend/src/css/custom.css`
- [x] T027 Implement graceful error handling (network issues, API errors) in Frontend
- [x] T028 [US3] Add rate limiting and input validation to Backend APIs (CANCELLED by user)
- [x] T029 Create deployment guide and verify build process in `README.md`

## Dependencies
- Phase 1 & 2 must be completed before starting Phase 3 (US1).
- Phase 3 (US1) is the MVP and prerequisite for Phase 4 (US2) and Phase 5 (US3).
- Phase 4 and Phase 5 can technically be developed in parallel, but sequential is recommended to minimize merge conflicts in `chat_service.py`.

## Implementation Strategy
- **MVP**: Complete Phase 1, 2, and 3. This provides a functional RAG chatbot.
- **Incremental**: Add conversational memory (Phase 4) and then text selection (Phase 5) as enhancements.
- **Documentation**: Use `specs/001-rag-chatbot/research.md` to document findings from OpenAI Agents SDK/ChatKit research to ensure correct usage of Context7-fetched info.
