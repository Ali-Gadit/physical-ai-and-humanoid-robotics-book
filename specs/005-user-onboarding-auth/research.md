# Research: User Onboarding and Authentication

**Feature**: User Onboarding and Authentication (`005-user-onboarding-auth`)
**Status**: Complete

## 1. Technical Context

### Unknowns & Resolutions

| Unknown | Resolution |
| :--- | :--- |
| **How to run Better Auth with Python Backend?** | Better Auth is TypeScript-only. We will create a separate Node.js service in the `user-auth/` directory (as requested) to host the Better Auth server. |
| **How to share Database?** | Both the Python backend and the `user-auth` service will connect to the same Neon PostgreSQL database (`NEON_DATABASE_URL`). Better Auth will manage the `user` and `session` tables. |
| **How does Python backend verify auth?** | The Python backend can verify authentication by either checking the session token in the database (since it shares the DB) or validating JWTs if configured. For simplicity and security, direct DB verification of the session token passed in headers/cookies is reliable. |
| **Where to store "Software/Hardware Background"?** | Better Auth allows extending the `User` schema or using a separate table. We will use Better Auth's schema extension capabilities or a linked `user_profiles` table in the same DB. |

## 2. Technology Choices

### Auth Service: Node.js + Better Auth
- **Decision**: Create a new Node.js service in `user-auth/`.
- **Rationale**: Better Auth is a TypeScript library. It cannot run directly inside the Python FastAPI app. A dedicated microservice is the standard pattern here.
- **Library**: `better-auth` (v1.x).

### Database: Neon (PostgreSQL)
- **Decision**: Use the existing Neon database.
- **Rationale**: User explicitly requested using the existing Neon DB. Better Auth supports Postgres out of the box.

### Frontend Integration: Better Auth Client
- **Decision**: Use `@better-auth/client` in the Docusaurus (React) app.
- **Rationale**: Provides hooks and utilities for seamless integration with the Better Auth server.

## 3. Architecture

1.  **`user-auth/` (Node.js Service)**:
    -   Runs Better Auth server.
    -   Exposes endpoints like `/api/auth/*`.
    -   Connects to Neon DB.
2.  **`backend/` (Python Service)**:
    -   Existing FastAPI app.
    -   Middleware to verify session tokens by querying the `session` table in Neon (or using a shared secret if JWT).
3.  **Frontend (Docusaurus)**:
    -   Connects to `user-auth` for Signup/Signin.
    -   Sends session token to `backend` for protected resources (like RAG chat).

## 4. Implementation Strategy

-   **Step 1**: Initialize `user-auth` service with TypeScript and Better Auth.
-   **Step 2**: Configure Better Auth with Neon DB and define the User schema (including new fields for background).
-   **Step 3**: Create Frontend pages (Signup/Signin) in Docusaurus using React components and Better Auth client.
-   **Step 4**: Implement Python middleware in `backend` to protect API routes using the shared session data.
