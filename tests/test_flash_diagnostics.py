"""Flash failures explain which extraction stage failed, without model calls."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import build_pdf


@pytest.fixture
def plain_pdf(tmp_path):
    path = tmp_path / "plain.pdf"
    path.write_bytes(build_pdf(["Alpha body", "", "Gamma body"]))
    return path


def test_page_fallback_keeps_diagnostics_before_early_return(plain_pdf, monkeypatch):
    from pageindex.flash import api

    monkeypatch.setattr(api, "FLAT_TREE_MAX_NODES", 2)
    result = api.page_index_flash(plain_pdf, summary=True, optimize="full")
    assert result["toc_source"] == "pages"
    assert len(result["structure"]) == 3
    assert "page_texts" not in result
    assert "optimize" not in result
    assert result["diagnostics"] == {
        "page_count": 3, "text_page_count": 2, "textless_pages": [2],
        "section_opener_count": 0, "assembled_heading_count": 0,
        "layout_status": "no_headings",
    }
    # No PDF text leaks into the shareable diagnosis.
    assert "Alpha" not in json.dumps(result["diagnostics"])
    reason = api.flash_rejection_reason(result)
    assert "2/3 pages contain extracted text" in reason
    assert "no headings survived outline assembly" in reason
    assert "mode='standard'" in reason


def test_unreadable_pdf_reports_zero_extracted_pages(tmp_path):
    from pageindex.flash import api

    path = tmp_path / "blank.pdf"
    path.write_bytes(build_pdf(["", ""]))
    result = api.page_index_flash(path, summary=False, optimize=False)
    assert result["toc_source"] == "unreadable"
    assert result["diagnostics"]["text_page_count"] == 0
    assert result["diagnostics"]["textless_pages"] == [1, 2]
    reason = api.flash_rejection_reason(result)
    assert "0/2 pages contain extracted text" in reason
    assert "OCR" in reason and "text encoding" in reason
    assert "standard" not in reason


@pytest.mark.parametrize("structured,status", [
    (True, "chapter_validation_failed"),
    (False, "heading_gap_too_large"),
])
def test_validation_failure_keeps_pre_rejection_heading_count(
        plain_pdf, monkeypatch, structured, status):
    from pageindex.flash import main, api

    outline = [SimpleNamespace(child_nodes=[SimpleNamespace(child_nodes=[])])]
    monkeypatch.setattr(main, "assemble_outline", lambda *a: outline)
    monkeypatch.setattr(main, "is_outline_valid", lambda *a: structured)
    monkeypatch.setattr(main, "is_chapter_outline_valid", lambda *a: False)
    monkeypatch.setattr(main, "mark_outline_block_types", lambda *a: None)
    monkeypatch.setattr(main, "compute_max_heading_gap", lambda *a: {"max_gap": 3})
    result = main.extract_toc(plain_pdf, workers=1, use_embedded_toc=False)
    assert result["structure"] == []
    assert result["diagnostics"]["assembled_heading_count"] == 2
    assert result["diagnostics"]["layout_status"] == status
    reason = api.flash_rejection_reason(result)
    assert "assembled headings=2" in reason
    assert ("chapter coverage validation" if structured else
            "gap between headings is too large") in reason


def test_bookmarks_can_rescue_layout_without_erasing_diagnostics(plain_pdf, monkeypatch):
    from pageindex.flash import main, embedded_toc, api

    monkeypatch.setattr(main, "assemble_outline", lambda *a: [])
    node = {"title": "Bookmark", "start_index": 1, "end_index": 3}
    monkeypatch.setattr(embedded_toc, "apply_embedded_toc",
                        lambda *a, **kw: ([node], "bookmarks"))
    result = api.page_index_flash(plain_pdf, summary=False, optimize=False)
    assert result["toc_source"] == "bookmarks"
    assert result["diagnostics"]["layout_status"] == "no_headings"
    assert api.flash_rejection_reason(result) is None


def test_local_sdk_exposes_diagnostics_on_refusal(plain_pdf, tmp_path, monkeypatch):
    from pageindex import PageIndexClient, PageIndexAPIError
    from pageindex.flash import api

    monkeypatch.setattr(api, "FLAT_TREE_MAX_NODES", 2)
    client = PageIndexClient(storage_path=str(tmp_path / "store"))
    with pytest.raises(PageIndexAPIError) as caught:
        client.submit_document(str(plain_pdf), mode="flash")
    assert "2/3 pages contain extracted text" in str(caught.value)
    assert "no headings survived outline assembly" in str(caught.value)


def test_cli_exposes_diagnostics_on_refusal(plain_pdf, tmp_path, monkeypatch):
    from pageindex.flash import api

    monkeypatch.setattr(api, "FLAT_TREE_MAX_NODES", 2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_pageindex.py", "--mode", "flash",
                                     "--pdf_path", str(plain_pdf)])
    script = Path(__file__).resolve().parents[1] / "run_pageindex.py"
    with pytest.raises(ValueError) as caught:
        runpy.run_path(str(script), run_name="__main__")
    assert "2/3 pages contain extracted text" in str(caught.value)
    assert "--mode standard" in str(caught.value)
    assert not (tmp_path / "results").exists()
