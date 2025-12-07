---
id: 0014
title: fix-openai-agents-package-name
stage: refactor
date: 2025-12-07
surface: agent
model: gemini-2.0-flash-exp
feature: 001-rag-chatbot
branch: 001-rag-chatbot
user: ali
command: null
labels: ["bug-fix", "requirements"]
links:
  spec: specs/001-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - backend/requirements.txt
tests:
  - null
---

## Prompt

bro in the requirement.txt you have write openai-agents-sdk but it is openai-agents

## Response snapshot

Corrected package name in `backend/requirements.txt` from `openai-agents-sdk` to `openai-agents`.

## Outcome

- ✅ Impact: Corrected dependency ensures proper installation of OpenAI Agents SDK.
- 🧪 Tests: None
- 📁 Files: 1 file modified.
- 🔁 Next prompts: User needs to re-run dependency installation and verify the UI.
- 🧠 Reflection: Important to double-check package names from documentation.

## Evaluation notes (flywheel)

- Failure modes observed: Incorrect package name in requirements.txt.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A