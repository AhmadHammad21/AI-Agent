import re
def remove_think_tags(text):
    """Removes text between <think>...</think> tags, including the tags themselves."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned_text = cleaned_text.lstrip("\n")  # Remove leading newlines
    return cleaned_text