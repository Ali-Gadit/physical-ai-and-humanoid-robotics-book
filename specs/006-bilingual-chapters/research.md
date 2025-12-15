# Research: Bilingual Chapters Implementation

**Decision**: Client-Side React Wrapper Component
**Rationale**:
- **Simplicity**: Keeps content co-located in `.mdx` files (English and Urdu side-by-side).
- **Performance**: Instant toggle without page reload.
- **Compatibility**: Works perfectly with Docusaurus static builds (requires `BrowserOnly` or `useEffect` for auth check).
- **Maintainability**: No complex i18n build configuration required for this specific "toggle" requirement.

**Alternatives Considered**:
- **Docusaurus i18n (Official)**: Rejected because it requires separate files/folders for translations and full site builds for each language. The requirement is a *dynamic toggle* for logged-in users on the *same* page.
- **Backend Content Serving**: Rejected to maintain static site performance and simplicity (hosting on GitHub Pages).

## Implementation Details

### 1. Component Architecture
We will create a `<BilingualChapter>` component that accepts two children (or slots):
- `english`: The English content (default).
- `urdu`: The Urdu content.

**Usage:**
```jsx
<BilingualChapter>
  <div className="english">...</div>
  <div className="urdu">...</div>
</BilingualChapter>
```
*Refinement*: To make writing easier, we might just look for children with specific props or order, but explicit props or named slots are safer.
*Better Usage (Props)*:
Passing large blocks of markdown as props is messy.
*Best Usage (Children)*:
```jsx
<BilingualChapter>
  <English>
     # Hello
  </English>
  <Urdu>
     # آداب
  </Urdu>
</BilingualChapter>
```
*Simplified usage for MVP*:
Just toggle visibility of `divs` based on a class or state.

### 2. State Management
- **Auth Check**: Use `better-auth` client SDK `authClient.getSession()`.
- **Persistence**: `localStorage.setItem('preferred_language', 'ur')`.

### 3. Styling
- Use CSS modules for isolation.
- Ensure RTL (Right-to-Left) support for Urdu container (`direction: rtl`).

## Unknowns Resolved
- **Auth Storage**: Confirmed `better-auth` uses cookies/local storage. We can check session availability client-side.
- **Docusaurus Compat**: `BrowserOnly` is standard for this.

## Action Plan
1.  Create `BilingualChapter` component.
2.  Implement `LanguageToggle` button logic.
3.  Add logic to hide/show children based on state.
