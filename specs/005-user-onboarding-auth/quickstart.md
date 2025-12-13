# Quickstart: User Onboarding and Authentication

## Prerequisites

- Node.js (v18+)
- Python 3.12+ (for backend)
- Neon Database URL

## Setup

### 1. User Auth Service (`user-auth/`)

1.  Navigate to the directory:
    ```bash
    cd user-auth
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Configure Environment:
    Create `.env` in `user-auth/`:
    ```env
    NEON_DATABASE_URL=postgres://...
    BETTER_AUTH_SECRET=... (generate one)
    BETTER_AUTH_URL=http://localhost:3000
    ```
4.  Run Migrations:
    ```bash
    npx @better-auth/cli migrate
    ```
5.  Start Dev Server:
    ```bash
    npm run dev
    ```

### 2. Frontend (`docusaurus-book/`)

1.  Navigate to directory:
    ```bash
    cd docusaurus-book
    ```
2.  Install Client:
    ```bash
    npm install better-auth
    ```
3.  Configure Env:
    Create `.env.local`:
    ```env
    BETTER_AUTH_URL=http://localhost:3000
    ```
4.  Start Docusaurus:
    ```bash
    npm start
    ```

### 3. Backend (`backend/`)

1.  Ensure `NEON_DATABASE_URL` matches `user-auth`.
2.  Start FastAPI:
    ```bash
    cd backend
    uv run api
    ```

## Verification

1.  Open `http://localhost:3000/api/auth/health` (or similar) to check Auth Service.
2.  Open Docusaurus at `http://localhost:3000` (or configured port).
3.  Click "Signup" in header.
4.  Fill form with Background Info.
5.  Check Database for new User record with correct `softwareSkillLevel` etc.
