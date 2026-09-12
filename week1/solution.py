"""
Day 2 homework exercise solution.

Upgrade the Day 1 website-summarizer project to use an open-source model
running locally via Ollama instead of a paid OpenAI API call.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1,
so we can keep using the `openai` Python client library - we just point it
at Ollama's base URL with a dummy API key instead of a real OpenAI one.

Run with uv from the week1 directory:
    uv run python solution.py <url>

Or interactively:
    uv run python solution.py
"""

import sys

from openai import OpenAI

from scraper import fetch_website_contents

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def messages_for(website: str) -> list[dict]:
    """Build the chat messages list to send to the LLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + website},
    ]


def summarize(url: str) -> str:
    """Fetch a website and summarize it using a local Ollama model."""
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    website = fetch_website_contents(url)
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=messages_for(website),
    )
    return response.choices[0].message.content


def main():
    """CLI entry point: pass a URL as an argument, or enter one when prompted."""
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter a URL to summarize: ")
    print("\nFetching and summarizing (via local Ollama model)...\n")
    print(summarize(url))


if __name__ == "__main__":
    main()
