# Quickstart Guide: RAG Chatbot for Physical AI and Humanoid Robotics Textbook

## Overview
This guide provides instructions for setting up and running the RAG chatbot system that integrates with the Physical AI and Humanoid Robotics textbook.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend development)
- Access to Google AI API (for Gemini embeddings and models)
- Access to OpenAI API (for OpenAI Agents SDK with custom model configuration)
- Qdrant Cloud account (free tier)
- Neon Serverless Postgres account

## Environment Setup

### 1. Clone and Initialize Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and service credentials
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your API endpoints
```

## Configuration

### Environment Variables

#### Backend (.env)
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key  # For OpenAI Agents SDK
GOOGLE_API_KEY=your_google_api_key  # For Gemini embeddings and models
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_HOST=your_qdrant_cluster_url

# Database
NEON_DATABASE_URL=your_neon_database_url

# Application
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

#### Frontend (.env)
```bash
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_SITE_URL=http://localhost:3000
```

## Running the System

### 1. Start Backend API
```bash
cd backend
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Frontend Development Server
```bash
cd frontend
npm start
```

### 3. Process Book Content
```bash
cd backend
source venv/bin/activate
python -m src.scripts.process_book_content
```

## API Endpoints

### Chat
- `POST /api/v1/chat` - Main chat interface
- `POST /api/v1/chat/text-selection` - Process selected text

### Content
- `GET /api/v1/documents` - List available documents
- `GET /api/v1/search` - Search book content

### Conversations
- `GET /api/v1/conversations` - List user conversations
- `GET /api/v1/conversations/{id}/messages` - Get conversation messages
- `DELETE /api/v1/conversations/{id}` - Delete conversation

## Frontend Integration

### Docusaurus Plugin
The chatbot integrates with Docusaurus through a custom plugin that adds:

1. Persistent chat button in bottom-left corner
2. Text selection context menu
3. Chat interface overlay

### Integration Files
- `src/components/ChatbotLauncher` - Main integration component
- `docusaurus.config.js` - Configuration for plugin inclusion

## Development Workflow

### Backend Development
1. Make changes to Python files in `src/`
2. API auto-reloads on file changes
3. Run tests: `pytest tests/`

### Frontend Development
1. Make changes to React components in `src/components/`
2. UI auto-updates on file changes
3. Run tests: `npm test`

## Testing

### Backend Tests
```bash
cd backend
source venv/bin/activate
pytest tests/unit/     # Unit tests
pytest tests/integration/  # Integration tests
```

### Frontend Tests
```bash
cd frontend
npm test  # Run Jest tests
npm run test:watch  # Run tests in watch mode
```

## Deployment

### Backend to Production
```bash
# Set production environment variables
export DEBUG=false
export API_HOST=0.0.0.0
export API_PORT=8000

# Deploy with your preferred platform (Heroku, AWS, etc.)
```

### Frontend Integration
The frontend components are built into the Docusaurus static site during the build process.

## Troubleshooting

### Common Issues
1. **API Keys**: Ensure all API keys are correctly set in environment variables
2. **CORS**: Check that frontend origin is allowed in backend CORS settings
3. **Vector Store**: Verify Qdrant connection and that book content has been processed

### Logs
- Backend: Check console output or configured log files
- Frontend: Check browser console for JavaScript errors

## Next Steps

1. Process your book content into the vector store
2. Test chat functionality with sample queries
3. Integrate with your Docusaurus book
4. Deploy to production environment