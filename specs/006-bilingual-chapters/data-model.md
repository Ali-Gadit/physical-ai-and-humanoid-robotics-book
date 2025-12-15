# Data Model: Bilingual Chapters

*Note: This feature is primarily client-side UI logic. No database schema changes are required.*

## Client-Side Entities

### User Preference (LocalStorage)

| Key | Value Type | Description |
|-----|------------|-------------|
| `bilingual_pref` | string (`'en'` \| `'ur'`) | The user's selected language. |

### Component Props (`BilingualChapter`)

| Prop | Type | Description |
|------|------|-------------|
| `children` | ReactNode[] | Expects exactly two children (divs) or specific structure. |

*Validation*: The component should gracefully handle missing Urdu content by falling back to English.
