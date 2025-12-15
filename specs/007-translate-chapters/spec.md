# Feature Specification: Translate Chapters

**Feature Branch**: `007-translate-chapters`  
**Created**: 2025-12-16  
**Status**: Draft  
**Input**: User description: "the toggle button and all the things are ready now we just need to translate all the chapters in urdu using the same logic as done in bilingual test thing"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Bilingual Content Integration (Priority: P1)

As a content consumer, I want all chapters in the textbook to be available in both English and Urdu so that I can read the material in my preferred language.

**Why this priority**: This is the core content delivery task that leverages the recently built bilingual toggle feature.

**Independent Test**: Can be tested by navigating to any chapter, logging in, and toggling the language.

**Acceptance Scenarios**:

1. **Given** I am a logged-in user viewing any chapter (e.g., Week 1, Module 1, etc.), **When** I click "Translate to Urdu", **Then** the English content is replaced by the corresponding Urdu content.
2. **Given** I am a logged-in user, **When** I view a chapter, **Then** the Urdu content matches the English content in meaning and structure (headings, lists, code blocks).
3. **Given** a chapter has special elements (images, code blocks), **When** viewed in Urdu, **Then** these elements are preserved or appropriately localized (e.g., right-to-left alignment for text, but code blocks remain LTR).

---

### User Story 2 - Consistent Formatting (Priority: P2)

As a reader, I want the Urdu content to be formatted correctly (Right-to-Left) so that it is readable and follows standard Urdu typography.

**Why this priority**: Ensures the quality and readability of the translated content.

**Independent Test**: Visually inspect Urdu content for RTL direction and proper font rendering.

**Acceptance Scenarios**:

1. **Given** I toggle to Urdu, **When** I look at the text, **Then** paragraphs are right-aligned and the text flows from right to left.
2. **Given** I see a list in Urdu, **When** I check the bullets/numbers, **Then** they appear on the right side of the item.

### Edge Cases

- **Partial Translations**: If a chapter is only partially translated during development, the `BilingualChapter` component must handle this gracefully (though the goal is 100% translation).
- **Code Blocks**: Code blocks usually remain in English (technical terms) and LTR. The wrapper must ensure code blocks inside Urdu sections don't get flipped to RTL incorrectly.
- **Mixed Content**: Sentences with mixed English terms (e.g., "ROS 2 use karein") must render correctly in the RTL context.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All existing Markdown (`.md` / `.mdx`) content files in `docusaurus-book/docs/` MUST be updated to use the `<BilingualChapter>` component.
- **FR-002**: For every paragraph, heading, and list in English, a corresponding Urdu translation MUST be provided within the `<div className="urdu">` slot.
- **FR-003**: The original English content MUST be preserved exactly within the `<div className="english">` slot.
- **FR-004**: Technical terms (e.g., "ROS 2", "Python", "C++", "Gazebo") SHOULD be kept in English or transliterated as appropriate for the technical context, but generally kept in English script for clarity.
- **FR-005**: Images and other media MUST be included in both English and Urdu sections if they are language-neutral, or localized versions if text exists in the image.

### Key Entities

- **Chapter Content**: The textual and media content of the book, now structured as a pair of (English, Urdu) nodes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the chapters in the `docs` folder are wrapped in `<BilingualChapter>` and contain Urdu text.
- **SC-002**: A logged-in user can successfully toggle language on every single page of the textbook.
- **SC-003**: Zero build errors in Docusaurus after converting all files to `.mdx` and adding imports.
