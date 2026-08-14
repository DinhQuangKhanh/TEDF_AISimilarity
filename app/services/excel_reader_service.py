from io import BytesIO

import pandas as pd

from app.utils.text_cleaner import clean_text, normalize_key

# A header cell is a short label; anything longer is prose and never a header.
MAX_HEADER_CELL_LENGTH = 50

HEADER_ALIASES = {
    "title": ["title", "title en", "english title", "project title", "tên đề tài", "ten de tai", "de tai"],
    "description": ["description", "summary", "mô tả", "mo ta"],
    "scope": ["scope", "phạm vi", "pham vi"],
    "objectives": ["objective", "objectives", "mục tiêu", "muc tieu"],
    "expected_result": ["expected result", "expected results", "expected output", "kết quả", "ket qua", "ket qua mong doi", "output"],
    "semester": ["semester", "học kỳ", "hoc ky"],
    "program": ["program", "ngành", "major", "chuong trinh"],
    "technologies": ["technology", "technologies", "tech stack", "công nghệ", "cong nghe"],
    "domains": ["domain", "field", "lĩnh vực", "linh vuc"],
}


class ExcelReaderService:
    def _match_field(self, value: str) -> str | None:
        """Which field a header cell names, or None.

        Header cells are short labels, never prose: a 1000-character description
        would otherwise match half the aliases by substring and outscore the real
        header row. Exact matches win over substring matches.
        """
        if not value or len(value) > MAX_HEADER_CELL_LENGTH:
            return None
        for field, aliases in HEADER_ALIASES.items():
            if any(alias == value for alias in aliases):
                return field
        for field, aliases in HEADER_ALIASES.items():
            if any(alias in value for alias in aliases):
                return field
        return None

    def _score_header_row(self, row_values: list[object]) -> int:
        # Count DISTINCT fields so one cell cannot inflate the score.
        fields = {
            field
            for value in row_values
            if (field := self._match_field(normalize_key(value))) is not None
        }
        return len(fields)

    def _detect_header(self, frame: pd.DataFrame) -> int:
        best_row = 0
        best_score = -1
        for idx in range(min(len(frame), 10)):
            score = self._score_header_row(frame.iloc[idx].tolist())
            if score > best_score:  # ties keep the earliest row
                best_score = score
                best_row = idx
        return best_row

    def _map_headers(self, headers: list[object]) -> dict[int, str]:
        mapped = {}
        for idx, header in enumerate(headers):
            field = self._match_field(normalize_key(header))
            if field:
                mapped[idx] = field
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
                if not raw.get("title"):
                    continue
                rows.append({"sheet_name": sheet_name, "row_number": row_number, "data": raw})
        return rows
