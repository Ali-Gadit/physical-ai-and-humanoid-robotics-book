# Quickstart: Translating a Chapter

## Step 1: Prepare the File
1.  Open the target `.md` or `.mdx` file.
2.  Ensure it has the `.mdx` extension (rename if necessary to support React components).

## Step 2: Add Import
Add this line after the frontmatter (the `---` block):
```jsx
import BilingualChapter from '@site/src/components/BilingualChapter';
```

## Step 3: Wrap and Translate
Refactor the content structure:

```jsx
---
title: My Chapter
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Introduction
    This is the original text.
  </div>
  <div className="urdu">
    # تعارف
    یہ اصل متن ہے۔
  </div>
</BilingualChapter>
```

## Step 4: Verify
1.  Run `npm start` in `docusaurus-book`.
2.  Navigate to the page.
3.  Login (if auth required) and toggle the language button.
