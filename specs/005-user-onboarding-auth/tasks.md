# Implementation Tasks: User Onboarding and Authentication

**Feature**: User Onboarding and Authentication (`005-user-onboarding-auth`)
**Spec**: [specs/005-user-onboarding-auth/spec.md](spec.md)
**Plan**: [specs/005-user-onboarding-auth/plan.md](plan.md)

## Phase 1: Setup

**Goal**: Initialize the new `user-auth` service and prepare the environment.

- [x] T001 Create `user-auth` directory and initialize Node.js TypeScript project (`npm init -y`, `tsc --init`) in `user-auth/`
- [x] T002 Install core dependencies (`better-auth`, `pg`, `dotenv`, `express` or `hono` for server) in `user-auth/package.json`
- [x] T003 Create `.env` file in `user-auth/` with `NEON_DATABASE_URL` and `BETTER_AUTH_SECRET` (placeholder) in `user-auth/.env`
- [x] T004 Create basic server entry point to expose Better Auth endpoints in `user-auth/src/index.ts`
- [x] T005 [P] Install `@better-auth/client` in Docusaurus project in `docusaurus-book/package.json`
- [x] T006 [P] Add `BETTER_AUTH_URL=http://localhost:3000` to `docusaurus-book/.env.local`

## Phase 2: Foundational

**Goal**: Configure Authentication logic and Database Schema.

- [x] T007 Configure Better Auth with Neon DB adapter and User schema extensions (software/hardware fields) in `user-auth/src/auth.ts`
- [x] T008 Generate and run database migrations to create User, Session, and Account tables in Neon DB via `user-auth/` CLI
- [x] T009 Create Auth Client helper for Docusaurus to initialize the client in `docusaurus-book/src/lib/auth-client.ts`
- [x] T010 [P] Implement Python database connection check in `backend/src/db/postgres.py` (ensure shared access)

## Phase 3: User Story 1 - New User Signup and Onboarding

**Goal**: Allow users to register and provide background information.
**Priority**: P1

- [x] T011 [US1] Create Signup page component with modern UI layout in `docusaurus-book/src/pages/signup.tsx`
- [x] T012 [US1] Implement registration form with name, email, password fields in `docusaurus-book/src/components/Auth/SignupForm.tsx`
- [x] T013 [US1] Add onboarding questions (Software Skills, OS, Hardware) to the signup form in `docusaurus-book/src/components/Auth/SignupForm.tsx`
- [x] T014 [US1] Integrate `authClient.signUp` call with additional fields in `docusaurus-book/src/components/Auth/SignupForm.tsx`
- [x] T015 [US1] Handle successful signup redirect and error display in `docusaurus-book/src/pages/signup.tsx`
- [x] T016 [US1] Verify user data (including profile fields) is correctly stored in Neon DB via `user-auth` logs or admin check

## Phase 4: User Story 2 - Existing User Signin

**Goal**: Allow existing users to sign in and access protected content.
**Priority**: P1

- [x] T017 [US2] Create Signin page component with modern UI in `docusaurus-book/src/pages/signin.tsx`
- [x] T018 [US2] Implement login form and integrate `authClient.signIn.email` in `docusaurus-book/src/components/Auth/SigninForm.tsx`
- [x] T019 [US2] Create generic Auth Guard component/hook to protect routes in `docusaurus-book/src/components/Auth/AuthGuard.tsx`
- [x] T020 [US2] [P] Create Python middleware to verify session token from cookies/headers against Neon DB in `backend/src/middleware/auth.py`
- [x] T021 [US2] Protect backend API routes (like Chat) using the new middleware in `backend/src/api/main.py`
- [x] T022 [US2] Implement Signout functionality in the header/navbar in `docusaurus-book/src/theme/NavbarItem/Component.tsx` (or custom Navbar)

## Phase 5: Polish & Cross-Cutting

**Goal**: Ensure a smooth user experience and high code quality.

- [x] T023 Add loading states and toast notifications for auth actions in `docusaurus-book/src/components/Auth/` components
- [x] T024 Ensure responsive design for Auth pages on mobile devices in `docusaurus-book/src/css/custom.css`
- [x] T025 [P] Add proper error messages for duplicate emails or invalid passwords
- [x] T026 Verify "Software/Hardware Background" data is accessible to the Python backend for RAG personalization (future hook)

## Dependencies

1.  **Setup (T001-T006)** must complete before **Foundational (T007-T010)**.
2.  **Foundational** must complete before **US1** or **US2**.
3.  **US1 (Signup)** and **US2 (Signin)** can be developed in parallel, but US1 is logical first step to create test users.
4.  **Backend Middleware (T020)** depends on **T008 (DB Schema)** being ready.

## Implementation Strategy

-   **MVP**: Focus on getting the `user-auth` service running and connecting to Neon. Then Build the Signup form.
-   **Parallelism**: Frontend UI (Docusaurus) and Backend Middleware (Python) can be built simultaneously once the DB schema is settled (Phase 2).
