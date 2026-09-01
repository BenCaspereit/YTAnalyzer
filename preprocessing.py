import emoji

def preprocess_text(text: str) -> str:
    # clean up text
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # changes emojis to text (example: ❤️ -> :red_heart:)
    text = emoji.demojize(text)
    
    # multiple spaces to single space
    text = " ".join(text.split())
    
    return text

def preprocess_batch(texts: list[str]) -> list[str]:
    return [preprocess_text(t) for t in texts]