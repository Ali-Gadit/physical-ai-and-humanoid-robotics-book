# Research: Translation Strategy

**Decision**: Manual/AI-Assisted Translation per File
**Rationale**:
- **Accuracy**: Automated bulk translation often fails on technical context (e.g., "ROS 2 node"). Processing files one-by-one allows for context-aware translation.
- **Component Wrapping**: Each file needs structural changes (imports, wrapping divs), which requires parsing the document structure.

**Alternatives Considered**:
- **Automated Script**: A script could wrap the content, but the quality of translation would be unknown without review. Given the scale (44 files), an iterative approach using the agent's capabilities is safer.

## Translation Guidelines

1.  **Imports**: Add `import BilingualChapter from '@site/src/components/BilingualChapter';` to the top of every file (after frontmatter).
2.  **Wrapping**:
    ```jsx
    <BilingualChapter>
      <div className="english">
        {Original Content}
      </div>
      <div className="urdu">
        {Translated Content}
      </div>
    </BilingualChapter>
    ```
3.  **Preservation**: Frontmatter MUST stay outside the `<BilingualChapter>` component.
4.  **Technical Terms**: Keep English terms (ROS 2, Python, etc.) in English script within the Urdu text for clarity.

## File List (Identified by Investigator)
The 44 files in `docusaurus-book/docs` will be processed in batches.
