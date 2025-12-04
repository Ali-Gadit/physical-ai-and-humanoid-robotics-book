---
id: 2
title: Update Project Constitution
stage: constitution
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: ["constitution"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:

---

## Prompt

'Project: Physical AI & Humanoid Robotics Textbook

Core principles:
- Textbook Creation: Create a textbook to teach a course in Physical AI & Humanoid Robotics.
- AI/Spec-Driven Development: Utilize Claude Code and Spec-Kit Plus for AI/Spec-Driven Book Creation.
- Docusaurus & GitHub Pages: The book will be written using Docusaurus and deployed to GitHub Pages.
- Integrated RAG Chatbot: Build and embed a Retrieval-Augmented Generation (RAG) chatbot within the published book. This chatbot, utilizing the OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres database, and Qdrant Cloud Free Tier, must be able to answer user questions about the book's content, including answering questions based only on text selected by the user.
- Reusable Intelligence (Bonus): Incorporate Claude Code Subagents and Agent Skills for reusable intelligence to earn bonus points.
- User Authentication & Personalization (Bonus): Implement Signup and Signin using Better-Auth.com. Collect user's software and hardware background at signup to personalize content.
- Content Personalization (Bonus): Allow logged users to personalize chapter content via a button at the start of each chapter.
- Content Translation (Bonus): Enable logged users to translate chapter content into Urdu in the chapters by pressing a button at the start of each chapter.

Key standards:
- Deliverables: A unified book project including AI/Spec-Driven Book Creation and Integrated RAG Chatbot Development.
- Technology Stack: Docusaurus, GitHub Pages, OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, Qdrant Cloud Free Tier, Better-Auth.com (for bonus features).
- Scoring: Base functionality (100 points), reusable intelligence (up to 50 bonus points), Signup/Signin & personalization (up to 50 bonus points), content personalization (up to 50 bonus points), Urdu translation (up to 50 bonus points).

Constraints:
- The project is part of a hackathon.
- The core deliverable is a textbook.
- Mandatory use of Claude Code and Spec-Kit Plus.

Success criteria:
- Successful deployment of the Docusaurus book to GitHub Pages.
- A fully functional RAG chatbot embedded in the book, capable of answering questions based on book content and user-selected text.
- Achieve maximum points by implementing all defined bonus features.'

## Response snapshot

New version: 1.0.0
Bump rationale: Minor version bump due to significant additions of principles, key standards, constraints, and success criteria.
Files flagged for manual follow-up:
- .specify/templates/plan-template.md
- .specify/templates/spec-template.md
- .specify/templates/tasks-template.md
Suggested commit message: `docs: amend constitution to v1.0.0 (principle additions + governance update)`

## Outcome

- ✅ Impact: Constitution updated with new project details.
- 🧪 Tests: none
- 📁 Files: .specify/memory/constitution.md
- 🔁 Next prompts: Review templates for consistency.
- 🧠 Reflection: Successfully updated the constitution based on user input and provided a clear summary.

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): none
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): none
