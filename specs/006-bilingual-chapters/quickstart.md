# Quickstart: Bilingual Chapters

## Overview
This feature allows logged-in users to toggle chapter content between English and Urdu.

## Usage for Content Creators

Wrap bilingual content in the `<BilingualChapter>` component. You must provide two clearly separated sections.

### Markdown Example

```jsx
import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    ## Introduction
    This is the English text.
  </div>
  <div className="urdu">
    ## تعارف
    یہ اردو متن ہے۔
  </div>
</BilingualChapter>
```

## Setup for Developers

1.  **Component Location**: `docusaurus-book/src/components/BilingualChapter/index.tsx`
2.  **State Logic**:
    - Checks `authClient.getSession()` on mount.
    - Reads/Writes `localStorage.getItem('bilingual_pref')`.
3.  **Styling**: Urdu content automatically gets `direction: rtl` and `text-align: right`.
