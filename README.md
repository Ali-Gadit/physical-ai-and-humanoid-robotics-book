# Physical AI & Humanoid Robotics Textbook with RAG Chatbot

This repository contains the code for an interactive textbook on Physical AI & Humanoid Robotics, augmented with a Retrieval-Augmented Generation (RAG) chatbot. The chatbot allows users to ask questions about the textbook content and receive context-aware answers.

## Project Overview

The project consists of two main parts:

1.  **Docusaurus Frontend:** The textbook content and the integrated chatbot UI.
2.  **FastAPI Backend:** Provides the RAG (Retrieval-Augmented Generation) capabilities, integrating with Qdrant for vector search and Neon Postgres for conversation history.

## Features

*   Interactive Q&A based on textbook content.
*   Context-aware conversations with persistent history (PostgreSQL).
*   Flexible AI agent powered by OpenAI Agents SDK (using Gemini models via LiteLLM).
*   Text selection for focused queries (currently disabled, but backend supports `quoted_text`).
*   Graceful error handling in the chatbot UI.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.10+** (for the backend)
*   **Node.js 18+** (for the frontend/Docusaurus)
*   **npm** (Node Package Manager, comes with Node.js)
*   **Git**
*   **Neon Postgres Database:** A connection string to a Neon Postgres database.
*   **Qdrant Cloud Free Tier:** A Qdrant host URL and API key.
*   **Google Gemini API Key:** An API key for accessing Google Gemini models.

## Local Development Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/physical-ai-and-humanoid-robotics-textbook.git
cd physical-ai-and-humanoid-robotics-textbook
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend/` directory with the following content. Replace the placeholder values with your actual keys and URLs.

```dotenv
# backend/.env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
QDRANT_API_KEY=YOUR_QDRANT_API_KEY
QDRANT_HOST=YOUR_QDRANT_HOST_URL # e.g., https://<cluster-id>.cloud.qdrant.io
NEON_DATABASE_URL="postgresql://YOUR_NEON_OWNER:YOUR_NEON_PASSWORD@YOUR_NEON_HOST/YOUR_NEON_DB_NAME?sslmode=require"
```
**Note**: Ensure `NEON_DATABASE_URL` uses `postgresql://` not `psql`.

### 3. Backend Setup

Navigate to the `backend/` directory and set up the Python environment.

```bash
cd backend/

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt

# Run database migrations (creates tables for chat history)
# This will be automatically handled when the server starts for the first time
# via init_db_engine in src/store/postgres_store.py
```

### 4. Ingest Book Content into Qdrant

You need to populate the Qdrant vector database with the textbook content. Ensure your virtual environment is active.

```bash
# From the project root directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend && python3 -m backend.scripts.ingest_content
```
This script will read the markdown files from `docusaurus-book/docs`, chunk them, generate Gemini embeddings, and upload them to your Qdrant instance. It will automatically create/recreate the Qdrant collection to match the embedding dimension.

### 5. Start the Backend Server

With the virtual environment active, run the FastAPI server:

```bash
# From the backend/ directory
uvicorn src.api.main:app --reload
```
The backend API will be available at `http://localhost:8000`.

### 6. Frontend Setup (Docusaurus)

Open a **new terminal window** and navigate to the `docusaurus-book/` directory.

```bash
cd docusaurus-book/

# Install Node.js dependencies
npm install

# Start the Docusaurus development server
npm start
```
The Docusaurus site (and chatbot UI) will be available at `http://localhost:3000`.

## Usage

1.  Open your browser to `http://localhost:3000`.
2.  You should see the chatbot button in the bottom-right corner.
3.  Click the button to open the chatbot UI.
4.  Ask questions related to the textbook content.
5.  Try the "Clear Chat" button to start a new conversation.

## Deployment Notes

For production deployment, consider the following:

*   **Environment Variables:** Use a secure method for managing environment variables (e.g., Kubernetes secrets, CI/CD variables).
*   **Backend:** Deploy the FastAPI application to a cloud provider (e.g., Render, Fly.io, Google Cloud Run) behind a Gunicorn/Uvicorn server.
*   **Frontend:** Deploy the Docusaurus build to a static site hosting service (e.g., GitHub Pages, Vercel, Netlify). The `npm run build` command generates static assets in the `build/` directory.
*   **Reverse Proxy:** Use a reverse proxy (like Nginx) to route requests from the frontend to the backend API if they are on different domains/ports.
*   **Database:** Ensure your Neon Postgres and Qdrant instances are accessible and secure from your deployed backend.

---
This `README.md` is a living document. Feel free to contribute!