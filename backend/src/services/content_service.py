import os
import re

def load_markdown_docs(docs_path: str = "docusaurus-book/docs"):
    """
    Loads all markdown content from the specified docusaurus docs path.
    Yields tuples of (file_path, content) for each markdown file.
    """
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".md") or file.endswith(".mdx"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                yield file_path, content

def clean_text(text: str) -> str:
    """Removes markdown syntax and extra whitespace from text."""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # Remove code blocks
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL) # Remove comments
    text = re.sub(r'\[.*?\]\(.*?\)','', text) # Remove links
    text = re.sub(r'#+\s*', '', text) # Remove headers
    text = re.sub(r'\*|_|\~', '', text) # Remove bold, italic, strikethrough
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text

if __name__ == "__main__":
    # Example usage:
    docs_generator = load_markdown_docs()
    for file_path, content in docs_generator:
        print(f"--- Loaded {file_path} ---")
        cleaned_content = clean_text(content)
        print(cleaned_content[:500]) # Print first 500 chars of cleaned content
        print("\n" + "="*80 + "\n")
