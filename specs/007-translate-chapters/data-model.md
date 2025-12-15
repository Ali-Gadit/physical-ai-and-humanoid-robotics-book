# Data Model: Content Structure

*No database changes. This describes the MDX content structure.*

## MDX Component Structure

| Node | Type | Description |
|------|------|-------------|
| `BilingualChapter` | Component | Root wrapper for the chapter content. |
| `div.english` | Element | Container for the original English Markdown. |
| `div.urdu` | Element | Container for the translated Urdu Markdown (RTL). |

## Frontmatter (Unchanged)

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Page title (used in sidebar and tab). |
| `sidebar_position` | number | Ordering in the sidebar. |
| `id` | string | (Optional) Custom route ID. |
