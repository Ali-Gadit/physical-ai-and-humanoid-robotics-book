# Implementation Plan: Integrated RAG Chatbot for Physical AI and Humanoid Robotics Textbook

**Branch**: `001-rag-chatbot` | **Date**: 2025-12-07 | **Spec**: [specs/001-rag-chatbot/spec.md](specs/001-rag-chatbot/spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Development of a Retrieval-Augmented Generation (RAG) chatbot that integrates with the Physical AI and Humanoid Robotics textbook. The system will utilize OpenAI Agents SDK and OpenAI ChatKit SDK (configured to use Gemini models), FastAPI, Neon Serverless Postgres database, and Qdrant Cloud Free Tier to enable users to ask questions about book content, with special functionality to answer questions based on user-selected text. The chatbot will be embedded in the book interface with a persistent chat button and context-aware text selection feature.

## Technical Context

**Language/Version**: Python 3.11 (for backend services), JavaScript/TypeScript (for frontend integration)
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, OpenAI ChatKit SDK, Google Generative AI SDK (for embeddings), Qdrant, Neon Postgres
**Storage**: Neon Serverless Postgres database, Qdrant vector database
**Testing**: pytest (backend), Jest (frontend)
**Target Platform**: Web-based textbook deployed on GitHub Pages
**Project Type**: Web application with embedded chatbot functionality
**Performance Goals**: Response times under 5 seconds for 95% of queries, support for 100 concurrent users
**Constraints**: <5 second p95 response time, integration with existing Docusaurus book structure, offline-capable book content
**Scale/Scope**: Single textbook with RAG functionality, support for 1000+ pages of technical content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification

1. **Principle IV (Integrated RAG Chatbot)**: ✅ Core requirement - Build and embed a RAG chatbot within the published book using OpenAI Agents/ChatKit SDKs (with Gemini models), FastAPI, Neon Serverless Postgres, and Qdrant Cloud Free Tier
2. **Principle I (Textbook Creation)**: ✅ Aligns with textbook delivery requirement
3. **Principle III (Docusaurus & GitHub Pages)**: ✅ Integration with existing Docusaurus book structure on GitHub Pages
4. **Technology Stack**: ✅ Uses specified technologies (FastAPI, Neon Postgres, Qdrant, OpenAI Agents/ChatKit SDKs with Gemini models)
5. **Functionality**: ✅ Enables answering questions about book content and questions based only on user-selected text
6. **Performance**: ✅ Meets response time requirements (<5 seconds) and concurrent user support (100 users)

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── rag/
│   │   ├── embedding.py      # Gemini embedding functionality
│   │   ├── vector_store.py   # Qdrant integration
│   │   └── retrieval.py      # RAG retrieval logic
│   ├── api/
│   │   ├── chat.py          # Chat API endpoints
│   │   └── text_selection.py # Text selection endpoints
│   ├── models/
│   │   ├── chat.py          # Chat models
│   │   └── content.py       # Content models
│   └── services/
│       ├── chat_service.py  # Chat business logic
│       └── content_service.py # Content processing logic
├── tests/
│   ├── unit/
│   └── integration/
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   ├── Chatbot.jsx     # Chatbot UI component
│   │   ├── TextSelection.jsx # Text selection handler
│   │   └── ChatButton.jsx  # Persistent chat button
│   └── services/
│       ├── api.js          # API client
│       └── textSelection.js # Text selection logic
├── package.json
└── docusaurus.config.js   # Docusaurus integration config

# Existing Docusaurus book structure
docs/
├── ...
└── ...

# Docusaurus integration
src/
├── components/
│   └── ChatbotLauncher/    # Integration with book UI
└── pages/
```

**Structure Decision**: Web application with separate backend (FastAPI) and frontend components. The backend handles RAG processing, vector storage, and API endpoints, while the frontend provides the chatbot UI and integrates with the existing Docusaurus book structure. This separation allows for independent scaling and maintenance of the RAG system from the textbook content.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No complexity violations identified - all constitution requirements are met with appropriate technical approach.*
