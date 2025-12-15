---

description: "Task list for Bilingual Chapters feature"
---

# Tasks: Bilingual Chapters

**Input**: Design documents from `/specs/006-bilingual-chapters/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, quickstart.md

**Tests**: Manual verification tasks included as per plan.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create feature directory structure in `docusaurus-book/src/components/BilingualChapter`
- [x] T002 [P] Create styles module file `docusaurus-book/src/components/BilingualChapter/styles.module.css` with basic RTL support classes

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Create `LanguageToggle.tsx` component skeleton in `docusaurus-book/src/components/BilingualChapter/LanguageToggle.tsx`
- [x] T004 Create `index.tsx` wrapper component skeleton in `docusaurus-book/src/components/BilingualChapter/index.tsx`
- [x] T005 [P] Create sample bilingual content MDX file for testing in `docusaurus-book/docs/test-bilingual.mdx`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 & 2 - Toggle Functionality (Priority: P1) 🎯 MVP

**Goal**: Logged-in users can switch between English and Urdu content.

**Independent Test**: Verify clicking the button switches content visibility in `test-bilingual.mdx`.

### Implementation for User Story 1 & 2

- [x] T006 [US1] Implement `better-auth` session check logic in `docusaurus-book/src/components/BilingualChapter/index.tsx` (using `<BrowserOnly>` or `useEffect`)
- [x] T007 [US1] Implement state management (useState) for language selection in `docusaurus-book/src/components/BilingualChapter/index.tsx`
- [x] T008 [US1] Implement toggle button UI and click handler in `docusaurus-book/src/components/BilingualChapter/LanguageToggle.tsx`
- [x] T009 [US1] Implement conditional rendering logic in `docusaurus-book/src/components/BilingualChapter/index.tsx` to show/hide children based on state
- [x] T010 [US2] Update `LanguageToggle.tsx` to dynamically show "Translate to English" vs "Translate to Urdu" text

**Checkpoint**: At this point, User Story 1 & 2 should be fully functional and testable independently

---

## Phase 4: User Story 3 - Persistent Language Preference (Priority: P2)

**Goal**: User's language choice is remembered across sessions/chapters.

**Independent Test**: Switch language, reload page, verify language remains selected.

### Implementation for User Story 3

- [x] T011 [US3] Add `localStorage` reading logic on component mount in `docusaurus-book/src/components/BilingualChapter/index.tsx`
- [x] T012 [US3] Add `localStorage` writing logic when language is toggled in `docusaurus-book/src/components/BilingualChapter/index.tsx`
- [x] T013 [US3] Handle edge case of missing `localStorage` (default to English)

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 5: User Story 4 - Guest User Experience (Priority: P2)

**Goal**: Guest users see English only and no toggle button.

**Independent Test**: Open `test-bilingual.mdx` in incognito window, verify no button appears.

### Implementation for User Story 4

- [x] T014 [US4] Refine auth check in `docusaurus-book/src/components/BilingualChapter/index.tsx` to strictly hide `LanguageToggle` if no session exists
- [x] T015 [US4] Ensure English content is rendered by default if auth check fails or is pending

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T016 [P] Apply final CSS styling for RTL text alignment in `docusaurus-book/src/components/BilingualChapter/styles.module.css`
- [x] T017 Verify feature in Docusaurus production build (run `npm run build` locally)
- [x] T018 Update `quickstart.md` with any API changes or usage notes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup
- **User Stories (Phase 3+)**: All depend on Foundational
  - US1 & US2 are coupled (toggle logic) and should be done together.
  - US3 (Persistence) depends on US1.
  - US4 (Guest) depends on US1 logic being in place.

### User Story Dependencies

- **US1 & US2**: Foundation -> Implementation
- **US3**: US1 -> Implementation
- **US4**: US1 -> Implementation

### Parallel Opportunities

- T002 (Styles) can run parallel to component logic.
- T005 (Test content) can run parallel to implementation.
- T011/T012 (Persistence) can theoretically be worked on by a second dev while first dev polishes UI, but best sequential here.

---

## Implementation Strategy

### MVP First (User Story 1 & 2)

1. Complete Setup & Foundation.
2. Build core `BilingualChapter` with state + toggle button.
3. Validate toggle works in-memory.

### Incremental Delivery

1. Add `localStorage` support (US3).
2. Refine Guest permissions (US4).
3. Final Polish (CSS, Build check).
