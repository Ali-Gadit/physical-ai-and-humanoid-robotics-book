# Feature Specification: Bilingual Chapters

**Feature Branch**: `006-bilingual-chapters`
**Created**: 2025-12-16
**Status**: Draft
**Input**: User description: "the logged user can translate the content in Urdu in the chapters by pressing a button at the start of each chapter. use the above discussion to know what it is and how it will be done"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Toggle to Urdu (Priority: P1)

As a logged-in user, I want to switch the chapter content to Urdu so that I can read it in my preferred language.

**Why this priority**: Core functionality of the feature.

**Independent Test**: Can be tested by logging in, navigating to a chapter, and clicking the translate button.

**Acceptance Scenarios**:

1. **Given** I am a logged-in user viewing a chapter in English, **When** I click "Translate to Urdu", **Then** the English text is hidden and Urdu text is displayed.
2. **Given** the content is in Urdu, **When** I look at the button, **Then** it says "Translate to English".

---

### User Story 2 - Toggle to English (Priority: P1)

As a logged-in user, I want to switch the chapter content back to English so that I can reference the original text.

**Why this priority**: Essential for bilingual usage and reverting changes.

**Independent Test**: Can be tested after switching to Urdu.

**Acceptance Scenarios**:

1. **Given** I am a logged-in user viewing a chapter in Urdu, **When** I click "Translate to English", **Then** the Urdu text is hidden and English text is displayed.
2. **Given** the content is in English, **When** I look at the button, **Then** it says "Translate to Urdu".

---

### User Story 3 - Persistent Language Preference (Priority: P2)

As a logged-in user, I want my language choice to be remembered so that I don't have to switch it on every new chapter.

**Why this priority**: Important for a smooth user experience (UX).

**Independent Test**: Can be tested by switching language, refreshing the page, or navigating to another chapter.

**Acceptance Scenarios**:

1. **Given** I have selected Urdu in a previous session or chapter, **When** I open a new chapter, **Then** it automatically loads in Urdu.
2. **Given** I have selected English, **When** I open a new chapter, **Then** it loads in English.

---

### User Story 4 - Guest User Experience (Priority: P2)

As a guest (not logged-in) user, I should only see English content and no translation options.

**Why this priority**: Ensures feature is exclusive to authenticated users as requested.

**Independent Test**: Can be tested in an incognito window or after logging out.

**Acceptance Scenarios**:

1. **Given** I am not logged in, **When** I view a chapter, **Then** I see English content.
2. **Given** I am not logged in, **When** I look at the top of the chapter, **Then** I do NOT see any translation button.

### Edge Cases

- **Authentication Expiry**: If a user's session expires while reading, the next page load should default to English and hide the button (graceful fallback).
- **Missing Translation**: If a specific chapter section lacks Urdu content, the English content should remain visible or a placeholder should be shown (implementation detail, but spec requires robustness). *Assumption: Content is manually managed in pairs.*
- **Slow Connection**: Ensure no "flash of incorrect language" or "flash of button" for guests if possible (though static site limitations apply).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST detect if a user is authenticated on the client-side (browser).
- **FR-002**: System MUST display a "Translate to Urdu" toggle button at the beginning of chapters ONLY for authenticated users.
- **FR-003**: System MUST hide the toggle button for unauthenticated (guest) users.
- **FR-004**: Clicking the toggle button MUST switch the visible content between English and Urdu versions instantly without reloading the page.
- **FR-005**: The toggle button label MUST dynamically update to reflect the target language (e.g., "Translate to Urdu" vs "Translate to English").
- **FR-006**: System MUST persist the user's language preference in the browser (e.g., LocalStorage).
- **FR-007**: System MUST automatically apply the persisted language preference when a user loads a chapter.
- **FR-008**: The default language for all users (if no preference is saved) MUST be English.

### Key Entities

- **User Session**: Represents the authentication state of the current visitor.
- **Bilingual Content Wrapper**: The container that holds both English and Urdu versions of the text.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Logged-in users can toggle between languages in under 200ms (perceived instant).
- **SC-002**: 100% of guest users see English content by default and cannot access the Urdu toggle.
- **SC-003**: User language preference persists across page reloads and navigation within the same browser session.