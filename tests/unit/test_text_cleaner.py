from app.utils.text_cleaner import clean_text, normalize_key, normalize_list


def test_clean_text():
    assert clean_text("  Hello\nWorld  ") == "Hello World"
    assert clean_text(None) is None


def test_normalize_key():
    assert normalize_key("Title EN") == "title en"


def test_normalize_list_aliases_and_dedupe():
    assert normalize_list(["ReactJS", "react", " Node JS "]) == ["React", "Node.js"]
