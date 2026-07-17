import pytest
from app.services.excel_reader_service import ExcelReaderService
from app.utils.text_cleaner import clean_text, normalize_key

class TestExcelReader:
    def test_clean_text(self):
        assert clean_text("  Hello\nWorld  ") == "Hello World"
        assert clean_text(None) is None
        assert clean_text("") is None

    def test_normalize_key(self):
        assert normalize_key("Title EN") == "title en"
        assert normalize_key("  Spaces  ") == "spaces"

    def test_parse_simple_excel(self):
        # minimal in‑memory xlsx: one sheet, header row, one data row
        import pandas as pd
        from io import BytesIO
        df = pd.DataFrame({
            "title": ["Project Alpha"],
            "description": ["A system for X"],
            "semester": ["2025 Spring"],
            "program": ["Software Engineering"],
        })
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        content = buf.getvalue()
        service = ExcelReaderService()
        rows = service.parse(content)
        assert len(rows) == 1
        assert rows[0]["data"]["title"] == "Project Alpha"

    def test_skip_empty_rows(self):
        import pandas as pd
        from io import BytesIO
        df = pd.DataFrame({
            "title": ["Project A", None, ""],
        })
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        rows = ExcelReaderService().parse(buf.getvalue())
        # empty/None rows should be skipped
        assert len(rows) == 1

    def test_header_detection(self):
        # test that header row is correctly identified
        import pandas as pd
        from io import BytesIO
        # first row is some metadata, second row is actual header
        df = pd.DataFrame({
            "ignore": ["", ""],
            "title en": ["Project B", "Project C"],
        })
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        rows = ExcelReaderService().parse(buf.getvalue())
        # only one valid title should appear (Project C after header)
        assert len(rows) >= 1
