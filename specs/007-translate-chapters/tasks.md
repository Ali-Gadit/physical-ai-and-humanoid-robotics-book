---

description: "Task list for translating chapters into Urdu"
---

# Tasks: Translate Chapters

**Input**: Design documents from `/specs/007-translate-chapters/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, quickstart.md

**Tests**: Manual verification of toggle and formatting for each batch.

**Organization**: Tasks are grouped by logical modules/weeks to manage the translation workload in batches.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

## Phase 1: Setup & Foundational Content

**Purpose**: Update the core landing and introductory pages.

- [x] T001 [US1] Translate `docusaurus-book/docs/index.md` (Home)
- [x] T002 [US1] Translate `docusaurus-book/docs/intro-physical-ai.md` (Intro)
- [x] T003 [US1] Translate `docusaurus-book/docs/hardware-requirements.md` (Requirements)
- [x] T004 [US1] Translate `docusaurus-book/docs/getting-started/installation-setup.md` (Installation)
- [x] T005 [US1] Translate `docusaurus-book/docs/configuration/course-setup.md` (Course Setup)

**Checkpoint**: Core navigation pages are bilingual. Verify toggle and layout.

---

## Phase 2: Module 1 - ROS 2 (Priority: P1)

**Purpose**: Translate the first technical module.

- [x] T006 [US1] Translate `docusaurus-book/docs/module1-ros2/overview.md`
- [x] T007 [US1] Translate `docusaurus-book/docs/module1-ros2/nodes-topics-services.md`
- [x] T008 [US1] Translate `docusaurus-book/docs/module1-ros2/python-ros-integration.md`
- [x] T009 [US1] Translate `docusaurus-book/docs/module1-ros2/urdf-humanoids.md`
- [x] T010 [US1] Translate `docusaurus-book/docs/module1-ros2/exercises.md`

**Checkpoint**: Module 1 is bilingual. Verify code block formatting.

---

## Phase 3: Module 2 - Digital Twin (Priority: P1)

**Purpose**: Translate simulation content.

- [x] T011 [US1] Translate `docusaurus-book/docs/module2-digital-twin/overview.md`
- [x] T012 [US1] Translate `docusaurus-book/docs/module2-digital-twin/gazebo-setup.md`
- [x] T013 [US1] Translate `docusaurus-book/docs/module2-digital-twin/physics-simulation.md`
- [x] T014 [US1] Translate `docusaurus-book/docs/module2-digital-twin/sensor-simulation.md`
- [x] T015 [US1] Translate `docusaurus-book/docs/module2-digital-twin/unity-integration.md`
- [x] T016 [US1] Translate `docusaurus-book/docs/module2-digital-twin/exercises.md`

**Checkpoint**: Module 2 is bilingual.

---

## Phase 4: Module 3 - AI Brain (Priority: P1)

**Purpose**: Translate AI and navigation content.

- [ ] T017 [US1] Translate `docusaurus-book/docs/module3-ai-brain/overview.md`
- [ ] T018 [US1] Translate `docusaurus-book/docs/module3-ai-brain/nav2-path-planning.md`
- [ ] T019 [US1] Translate `docusaurus-book/docs/module3-ai-brain/isaac-sim.md`
- [ ] T020 [US1] Translate `docusaurus-book/docs/module3-ai-brain/isaac-ros.md`
- [ ] T021 [US1] Translate `docusaurus-book/docs/module3-ai-brain/exercises.md`

**Checkpoint**: Module 3 is bilingual.

---

## Phase 5: Module 4 - VLA (Priority: P1)

**Purpose**: Translate Vision-Language-Action content.

- [ ] T022 [US1] Translate `docusaurus-book/docs/module4-vla/overview.md`
- [ ] T023 [US1] Translate `docusaurus-book/docs/module4-vla/cognitive-planning.md`
- [ ] T024 [US1] Translate `docusaurus-book/docs/module4-vla/voice-to-action.md`
- [ ] T025 [US1] Translate `docusaurus-book/docs/module4-vla/capstone-project.md`
- [ ] T026 [US1] Translate `docusaurus-book/docs/module4-vla/exercises.md`

**Checkpoint**: Module 4 is bilingual.

---

## Phase 6: Weekly Guides (Priority: P2)

**Purpose**: Translate weekly breakdown guides.

- [x] T027 [US1] Translate `docusaurus-book/docs/week1-2-intro.md`
- [x] T028 [US1] Translate `docusaurus-book/docs/week3-5-ros2.md`
- [x] T029 [US1] Translate `docusaurus-book/docs/week6-7-gazebo.md`
- [x] T030 [US1] Translate `docusaurus-book/docs/week11-12-humanoid.md`
- [x] T031 [US1] Translate `docusaurus-book/docs/week13-conversational-robotics.md`
- [x] T032 [US1] Translate `docusaurus-book/docs/weekly/weeks1-2-intro-physical-ai.md`
- [x] T033 [US1] Translate `docusaurus-book/docs/weekly/weeks3-5-ros2.md`
- [x] T034 [US1] Translate `docusaurus-book/docs/weekly/weeks6-7-gazebo.md`
- [x] T035 [US1] Translate `docusaurus-book/docs/weekly/weeks8-10-isaac.md`
- [x] T036 [US1] Translate `docusaurus-book/docs/weekly/weeks11-12-humanoid.md`
- [x] T037 [US1] Translate `docusaurus-book/docs/weekly/week13-conversational-robotics.md`

**Checkpoint**: Weekly guides are bilingual.

---

## Phase 7: Support & Validation (Priority: P2)

**Purpose**: Translate supporting documentation.

- [x] T038 [US1] Translate `docusaurus-book/docs/support/learning-outcomes.md`
- [x] T039 [US1] Translate `docusaurus-book/docs/support/best-practices.md`
- [x] T040 [US1] Translate `docusaurus-book/docs/support/troubleshooting.md`
- [x] T041 [US1] Translate `docusaurus-book/docs/support/assessments.md`
- [x] T042 [US1] Translate `docusaurus-book/docs/validation/validation-testing.md`
- [x] T043 [US1] Translate `docusaurus-book/docs/reference/glossary.md`

**Checkpoint**: All documentation files are translated.

---

## Phase 8: Final Validation

- [ ] T044 Verify Docusaurus build (`npm run build`)
- [ ] T045 Spot check random pages for correct RTL rendering and toggle persistence.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1-7**: Can be executed sequentially or in parallel if multiple agents/translators are available.
- **Phase 8**: Requires all previous phases to be complete.

### User Story Dependencies

- **US1 (Content)**: All tasks T001-T043 directly implement this.
- **US2 (Formatting)**: Implicitly handled by the `<BilingualChapter>` component and proper Urdu text entry during T001-T043.

---

## Implementation Strategy

### Sequential Batching

1. Complete Phase 1 (Setup) to ensure the core pages look good.
2. Proceed module by module.
3. Validate after each module to catch any recurring formatting issues early.

### Handling Translation

For each file task:
1. Read the English content.
2. Add the `import` statement.
3. Wrap content in `<BilingualChapter>`.
4. Create the Urdu translation block, preserving English technical terms.
