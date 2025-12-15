# Implementation Plan: Translate Chapters

**Branch**: `007-translate-chapters` | **Date**: 2025-12-16 | **Spec**: [specs/007-translate-chapters/spec.md](../spec.md)
**Input**: Feature specification from `specs/007-translate-chapters/spec.md`

## Summary

This feature involves updating all 44 existing Markdown/MDX files in `docusaurus-book/docs` to use the `<BilingualChapter>` component. Each file will be refactored to wrap the existing English content in a `<div className="english">` block and add a new `<div className="urdu">` block containing the Urdu translation.

## Technical Context

**Language/Version**: Markdown/MDX, Urdu text
**Primary Dependencies**: `docusaurus-book/src/components/BilingualChapter`
**Storage**: N/A (Content files)
**Testing**: Manual verification of toggle and formatting
**Target Platform**: Web (Docusaurus static site)
**Project Type**: Content Update
**Performance Goals**: N/A
**Constraints**: Must preserve all existing English content exactly. Urdu translation must use correct RTL formatting.
**Scale/Scope**: 44 files to update.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **VIII. Content Translation (Bonus)**: Directly fulfills this bonus objective.
- **I. Textbook Creation**: Enhances the textbook.

**Gate Status**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/007-translate-chapters/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (N/A)
├── quickstart.md        # Phase 1 output
└── contracts/           # Phase 1 output (N/A)
```

### Source Code

```text
docusaurus-book/docs/
├── [all .md/.mdx files] # Target for update
```

**Structure Decision**: N/A - modifying existing content files.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
