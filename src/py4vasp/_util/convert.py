def text_to_string(text):
    "Text can be either bytes or string"
    try:
        return text.decode()
    except (UnicodeDecodeError, AttributeError):
        return text
