import re
import unicodedata

TECH_ALIASES = {
    "reactjs": "React",
    "react js": "React",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
}


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_key(value: object) -> str:
    text = clean_text(value) or ""
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹぁ-んァ-ン一-龯]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        text = clean_text(value)
        if not text:
            continue
        alias = TECH_ALIASES.get(text.lower(), text)
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            result.append(alias)
    return result


def tokenize(value: str | None) -> set[str]:
    key = normalize_key(value)
    return {token for token in key.split() if len(token) > 1}
