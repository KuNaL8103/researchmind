from types import SimpleNamespace

from src.tools import collect_urls_from_messages, extract_urls


def test_extract_urls_deduplicates_and_strips_trailing_punctuation():
    text = (
        "See https://example.com/a, and https://example.com/a. "
        "Also **https://example.com/b** and http://example.com/a/."
    )
    assert extract_urls(text) == ["https://example.com/a", "https://example.com/b"]


def test_collect_urls_from_all_message_types():
    messages = [
        SimpleNamespace(content="No URLs in the final answer."),
        {"content": "URL: https://example.com/tool-a\nURL: https://example.com/tool-b"},
        SimpleNamespace(
            content=[
                {"type": "text", "text": "https://example.com/tool-c"},
                {"type": "tool_use", "name": "search"},
            ]
        ),
        {"content": "https://example.com/tool-a"},
    ]

    urls = collect_urls_from_messages(messages)

    assert urls == [
        "https://example.com/tool-a",
        "https://example.com/tool-b",
        "https://example.com/tool-c",
    ]
