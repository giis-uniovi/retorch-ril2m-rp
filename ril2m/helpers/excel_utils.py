# -*- coding: utf-8 -*-
"""Excel report generation utilities (requires openpyxl)."""

from typing import Any

_SHEET_SUMMARY = "Summary"
_SHEET_FOLD = "Fold Details"
_SHEET_RUN = "Run Details"


def write_metrics_excel(
    all_results: list[dict],
    output_path: str,
    summary_fields: list[str],
    fold_fields: list[str],
    run_fields: list[str] | None = None,
) -> None:
    """Write a multi-sheet Excel workbook for cross-validation metrics.

    Sheets written:
    - ``Summary``      – one row per SUT, columns from *summary_fields*.
    - ``Fold Details`` – one row per fold (averaged across runs), from *fold_fields*.
    - ``Run Details``  – one row per fold × run, from *run_fields* (omitted if ``None``).

    Parameters
    ----------
    all_results:
        List of per-SUT result dicts as returned by ``compute_sut_metrics``.
    output_path:
        Full path for the ``.xlsx`` file to create.
    summary_fields:
        Column names for the Summary sheet.
    fold_fields:
        Column names for the Fold Details sheet (``"sut"`` injected automatically).
    run_fields:
        Column names for the Run Details sheet.  Pass ``None`` to omit the sheet.
    """
    import openpyxl  # noqa: PLC0415
    from openpyxl.styles import Alignment, Font, PatternFill  # noqa: PLC0415

    styles = {
        "header_fill": PatternFill("solid", fgColor="1F4E79"),
        "header_font": Font(bold=True, color="FFFFFF"),
        "alignment": Alignment(horizontal="center", wrap_text=True),
        "alt_fill": PatternFill("solid", fgColor="D6E4F0"),
    }

    wb = openpyxl.Workbook()
    _write_summary_sheet(wb.active, all_results, summary_fields, styles)
    _write_fold_sheet(wb.create_sheet(_SHEET_FOLD), all_results, fold_fields, styles)
    if run_fields is not None:
        _write_run_sheet(wb.create_sheet(_SHEET_RUN), all_results, run_fields, styles)
    wb.save(output_path)


# ── private sheet builders ────────────────────────────────────────────────────

def _fmt(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def _header_row(ws, fields: list[str], styles: dict) -> None:
    ws.append(fields)
    for cell in ws[1]:
        cell.fill = styles["header_fill"]
        cell.font = styles["header_font"]
        cell.alignment = styles["alignment"]
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 14


def _write_summary_sheet(ws, all_results: list[dict], fields: list[str], styles: dict) -> None:
    ws.title = "Summary"
    _header_row(ws, fields, styles)
    for i, r in enumerate(all_results, start=2):
        if r.get("n_folds", 0) == 0:
            continue
        ws.append([_fmt(r.get(f)) for f in fields])
        if i % 2 == 0:
            for cell in ws[i]:
                cell.fill = styles["alt_fill"]


def _write_fold_sheet(ws, all_results: list[dict], fields: list[str], styles: dict) -> None:
    _header_row(ws, fields, styles)
    for row_idx, (r, fold) in enumerate(
        ((r, fold) for r in all_results for fold in r.get("fold_details", [])),
        start=2,
    ):
        ws.append([_fmt({"sut": r["sut"], **fold}.get(f)) for f in fields])
        if row_idx % 2 == 0:
            for cell in ws[row_idx]:
                cell.fill = styles["alt_fill"]


# ── incremental Excel helpers ─────────────────────────────────────────────────

def _make_styles():
    from openpyxl.styles import Alignment, Font, PatternFill  # noqa: PLC0415
    return {
        "header_fill": PatternFill("solid", fgColor="1F4E79"),
        "header_font": Font(bold=True, color="FFFFFF"),
        "alignment": Alignment(horizontal="center", wrap_text=True),
        "alt_fill": PatternFill("solid", fgColor="D6E4F0"),
    }


def init_metrics_excel(
    path: str,
    summary_fields: list[str],
    fold_fields: list[str],
    run_fields: list[str],
) -> None:
    """Create a fresh Excel workbook with headers on all three sheets.

    Call once before the first ``append_fold_to_excel`` for a given file.
    """
    import openpyxl  # noqa: PLC0415

    styles = _make_styles()
    wb = openpyxl.Workbook()
    ws_sum = wb.active
    ws_sum.title = "Summary"
    _header_row(ws_sum, summary_fields, styles)
    _header_row(wb.create_sheet(_SHEET_FOLD), fold_fields, styles)
    _header_row(wb.create_sheet(_SHEET_RUN), run_fields, styles)
    wb.save(path)


def append_fold_to_excel(
    path: str,
    sut: str,
    fold_entry: dict,
    fold_fields: list[str],
    run_fields: list[str],
) -> None:
    """Load the existing Excel, append one fold row and all its run rows, and save.

    Alternating row fill is applied based on the current row count in each sheet.
    """
    import openpyxl  # noqa: PLC0415
    from openpyxl.styles import PatternFill  # noqa: PLC0415

    alt_fill = PatternFill("solid", fgColor="D6E4F0")
    wb = openpyxl.load_workbook(path)

    # Fold Details sheet
    ws_fold = wb[_SHEET_FOLD]
    fold_row = {"sut": sut, **fold_entry}
    ws_fold.append([_fmt(fold_row.get(f)) for f in fold_fields])
    if ws_fold.max_row % 2 == 0:
        for cell in ws_fold[ws_fold.max_row]:
            cell.fill = alt_fill

    # Run Details sheet
    ws_run = wb[_SHEET_RUN]
    for run in fold_entry.get("runs", []):
        run_row = {
            "sut": sut,
            "fold_id": fold_entry["fold_id"],
            "testname": fold_entry["testname"],
            **run,
        }
        ws_run.append([_fmt(run_row.get(f)) for f in run_fields])
        if ws_run.max_row % 2 == 0:
            for cell in ws_run[ws_run.max_row]:
                cell.fill = alt_fill

    wb.save(path)


def update_summary_in_excel(
    path: str,
    sut_summaries: list[dict],
    summary_fields: list[str],
) -> None:
    """Rebuild the Summary sheet from *sut_summaries* (one row per completed SUT).

    The Fold Details and Run Details sheets are left unchanged.
    """
    import openpyxl  # noqa: PLC0415
    from openpyxl.styles import PatternFill  # noqa: PLC0415

    alt_fill = PatternFill("solid", fgColor="D6E4F0")
    wb = openpyxl.load_workbook(path)

    # Replace Summary sheet content (delete rows after header, rewrite)
    ws = wb["Summary"]
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.value = None

    for i, summary in enumerate(sut_summaries, start=2):
        for j, field in enumerate(summary_fields, start=1):
            ws.cell(row=i, column=j, value=_fmt(summary.get(field)))
        if i % 2 == 0:
            for cell in ws[i]:
                cell.fill = alt_fill

    wb.save(path)


# ── global workbook (cross-experiment) ───────────────────────────────────────

_SHEET_ALL = "All Results"


def _append_rows_to_sheet(ws, rows: list[dict], fields: list[str], alt_fill) -> None:
    """Append *rows* to *ws*, applying alternating row fill."""
    for i, row in enumerate(rows, start=ws.max_row + 1):
        ws.append([_fmt(row.get(f)) for f in fields])
        if i % 2 == 0:
            for cell in ws[i]:
                cell.fill = alt_fill


def init_global_metrics_excel(path: str, summary_fields: list[str], fold_fields: list[str]) -> None:
    """Create the global Excel workbook with Summary and All-Results sheets.

    Called once before the first ``append_fold_to_global_excel`` for a given file.
    """
    import openpyxl  # noqa: PLC0415

    styles = _make_styles()
    wb = openpyxl.Workbook()
    ws_sum = wb.active
    ws_sum.title = _SHEET_SUMMARY
    _header_row(ws_sum, summary_fields, styles)
    _header_row(wb.create_sheet(_SHEET_ALL), fold_fields, styles)
    wb.save(path)


def append_fold_to_global_excel(
    path: str,
    model_name: str,
    fold_row: dict,
    fold_fields: list[str],
) -> None:
    """Append one fold row to the All-Results sheet and the model-specific sheet.

    If the model sheet does not yet exist it is created with the same header.
    Both the All-Results and the model sheet are updated in a single save.
    """
    import openpyxl  # noqa: PLC0415

    styles = _make_styles()
    wb = openpyxl.load_workbook(path)

    # All Results
    ws_all = wb[_SHEET_ALL]
    ws_all.append([_fmt(fold_row.get(f)) for f in fold_fields])
    if ws_all.max_row % 2 == 0:
        for cell in ws_all[ws_all.max_row]:
            cell.fill = styles["alt_fill"]

    # Per-model sheet (create on first encounter)
    sheet_name = model_name[:31]          # Excel sheet name limit
    if sheet_name not in wb.sheetnames:
        ws_model = wb.create_sheet(sheet_name)
        _header_row(ws_model, fold_fields, styles)
    else:
        ws_model = wb[sheet_name]
    ws_model.append([_fmt(fold_row.get(f)) for f in fold_fields])
    if ws_model.max_row % 2 == 0:
        for cell in ws_model[ws_model.max_row]:
            cell.fill = styles["alt_fill"]

    wb.save(path)


def update_global_summary_in_excel(
    path: str,
    summary_rows: list[dict],
    summary_fields: list[str],
) -> None:
    """Rebuild the Summary sheet in the global Excel from *summary_rows*.

    All other sheets are left unchanged.
    """
    import openpyxl  # noqa: PLC0415
    from openpyxl.styles import PatternFill  # noqa: PLC0415

    alt_fill = PatternFill("solid", fgColor="D6E4F0")
    wb = openpyxl.load_workbook(path)
    ws = wb[_SHEET_SUMMARY]

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.value = None

    for i, row in enumerate(summary_rows, start=2):
        for j, field in enumerate(summary_fields, start=1):
            ws.cell(row=i, column=j, value=_fmt(row.get(field)))
        if i % 2 == 0:
            for cell in ws[i]:
                cell.fill = alt_fill

    wb.save(path)


def _write_run_sheet(ws, all_results: list[dict], fields: list[str], styles: dict) -> None:
    _header_row(ws, fields, styles)
    row_idx = 2
    for r in all_results:
        for fold in r.get("fold_details", []):
            for run in fold.get("runs", []):
                row = {
                    "sut": r["sut"],
                    "fold_id": fold["fold_id"],
                    "testname": fold["testname"],
                    **run,
                }
                ws.append([_fmt(row.get(f)) for f in fields])
                if row_idx % 2 == 0:
                    for cell in ws[row_idx]:
                        cell.fill = styles["alt_fill"]
                row_idx += 1
