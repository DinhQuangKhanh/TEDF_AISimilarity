from io import BytesIO

import pandas as pd

from app.utils.text_cleaner import clean_text, normalize_key

HEADER_ALIASES = {
    "title_en": ["title en", "english title", "project title", "title", "ten de tai en"],
    "title_vn": ["title vn", "vietnamese title", "tên đề tài", "de tai", "ten de tai"],
    "description": ["description", "summary", "mô tả", "mo ta"],
    "scope": ["scope", "phạm vi", "pham vi"],
    "semester": ["semester", "học kỳ", "hoc ky"],
    "program": ["program", "ngành", "major", "chuong trinh"],
    "technologies": ["technology", "technologies", "tech stack", "công nghệ", "cong nghe"],
    "domains": ["domain", "field", "lĩnh vực", "linh vuc"],
}


class ExcelReaderService:
    def _score_header_row(self, row_values: list[object]) -> int:
        normalized = [normalize_key(value) for value in row_values]
        score = 0
        for value in normalized:
            for aliases in HEADER_ALIASES.values():
                if any(alias in value for alias in aliases):
                    score += 1
        return score

    def _detect_header(self, frame: pd.DataFrame) -> int:
        best_row = 0
        best_score = -1
        for idx in range(min(len(frame), 10)):
            score = self._score_header_row(frame.iloc[idx].tolist())
            if score > best_score:
                best_score = score
                best_row = idx
        return best_row

    def _map_headers(self, headers: list[object]) -> dict[int, str]:
        mapped = {}
        for idx, header in enumerate(headers):
            key = normalize_key(header)
            for field, aliases in HEADER_ALIASES.items():
                if any(alias in key for alias in aliases):
                    mapped[idx] = field
                    break
        return mapped

    def parse(self, content: bytes) -> list[dict]:
        workbook = pd.read_excel(BytesIO(content), sheet_name=None, header=None)
        rows: list[dict] = []
        for sheet_name, frame in workbook.items():
            if frame.empty:
                continue
            header_idx = self._detect_header(frame)
            headers = frame.iloc[header_idx].tolist()
            mapping = self._map_headers(headers)
            data = frame.iloc[header_idx + 1 :]
            for row_number, (_, row) in enumerate(data.iterrows(), start=header_idx + 2):
                raw = {mapping[idx]: clean_text(value) for idx, value in enumerate(row.tolist()) if idx in mapping}
                raw = {key: value for key, value in raw.items() if value}
                if not raw:
                    continue
                if not raw.get("title_en") and not raw.get("title_vn"):
                    continue
                rows.append({"sheet_name": sheet_name, "row_number": row_number, "data": raw})
        return rows
