# Implementation Plan: Bilingual Chapters

**Branch**: `006-bilingual-chapters` | **Date**: 2025-12-16 | **Spec**: [specs/006-bilingual-chapters/spec.md](../spec.md)
**Input**: Feature specification from `specs/006-bilingual-chapters/spec.md`

## Summary

This feature enables authenticated users to toggle chapter content between English and Urdu using a client-side button. It leverages a custom React component (`<BilingualChapter>`) within Docusaurus MDX files to wrap both versions of the content. The toggle state is managed via browser `localStorage` and `better-auth` session detection.

## Technical Context

**Language/Version**: TypeScript 5.x (Docusaurus v3), React 18
**Primary Dependencies**: `better-auth` (client client), Docusaurus core
**Storage**: Browser `localStorage` for language preference
**Testing**: Manual verification (Docusaurus static site constraints)
**Target Platform**: Web (Docusaurus static site)
**Project Type**: Docusaurus Plugin / React Component
**Performance Goals**: Instant toggle (<200ms), no layout shift
**Constraints**: Must work with static site generation (SSG) - requires `<BrowserOnly>` or `useEffect` for auth checks.
**Scale/Scope**: Component used across multiple chapters.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Textbook Creation**: Enhances textbook accessibility.
- **II. AI/Spec-Driven Development**: Follows spec workflow.
- **III. Docusaurus & GitHub Pages**: Fully compatible with Docusaurus architecture.
- **VI. User Authentication & Personalization**: Directly uses auth for personalization.
- **VIII. Content Translation (Bonus)**: Directly implements this bonus objective.

**Gate Status**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/006-bilingual-chapters/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (N/A - purely client-side state)
├── quickstart.md        # Phase 1 output
└── contracts/           # Phase 1 output (N/A - no API changes)
```

### Source Code

```text
docusaurus-book/src/
├── components/
│   └── BilingualChapter/
│       ├── index.tsx          # Main wrapper component
│       ├── LanguageToggle.tsx # The button UI
│       └── styles.module.css  # Scoped styles
├── theme/
│   └── Root.js                # (Optional) Global state wrapper if needed
```

**Structure Decision**: Option 2 (Web application - tailored for Docusaurus)

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |