#!/usr/bin/env python3
"""Rebuild the H100 research PDF with the current audited width-dispatch data.

Requires reportlab and pypdf. The report keeps the original 17-page historical
record from commit a74c073, then adds a current status page, a matrix result page,
and a resource-monitoring page derived from the active checkpoints and partial audit.
"""

from __future__ import annotations

import io
import hashlib
import json
import statistics
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "a74c073"
BASELINE_PDF = "output/pdf/H100_RMSNorm_연구누적기록_2026-11-27.pdf"
OUTPUT_PDF = ROOT / "output/pdf/H100_RMSNorm_연구누적기록.pdf"
RAW_DIR = ROOT / "results/width-dispatch/raw"
CHECKPOINT_DIR = RAW_DIR / "checkpoints"
AUDIT_PATH = RAW_DIR / "partial-audit.json"
PIN_PATH = RAW_DIR / "source-pin.json"

NAVY = colors.HexColor("#10263F")
INK = colors.HexColor("#1D2B3A")
TEAL = colors.HexColor("#168A8A")
BLUE_PALE = colors.HexColor("#EEF3F8")
GOLD_PALE = colors.HexColor("#FFF5E5")
RED_PALE = colors.HexColor("#FCEEEE")
MUTED = colors.HexColor("#5F6F7E")
GRID = colors.HexColor("#D6E0E8")
WHITE = colors.white

BACKENDS = (
    ("triton_4w_1r", "4w/1r"),
    ("triton_8w_1r", "8w/1r"),
    ("triton_4w_4r", "4w/4r"),
    ("torch_compile", "compile"),
)


def font_path() -> Path:
    candidates = (
        Path.home() / "Library/Fonts/NotoSansKR-Regular.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("Noto Sans KR/CJK font not found; install a Korean-capable TrueType font.")


def styles():
    pdfmetrics.registerFont(TTFont("NotoKR", str(font_path())))
    base = getSampleStyleSheet()
    return {
        "section": ParagraphStyle("section", parent=base["Heading1"], fontName="NotoKR",
                                  fontSize=14, leading=19, textColor=NAVY, spaceAfter=6),
        "subsection": ParagraphStyle("subsection", parent=base["Heading2"], fontName="NotoKR",
                                     fontSize=10, leading=14, textColor=TEAL, spaceBefore=5, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="NotoKR", fontSize=8,
                                leading=12, textColor=INK, spaceAfter=5, wordWrap="CJK"),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="NotoKR", fontSize=6.8,
                                 leading=9.5, textColor=MUTED, spaceAfter=2, wordWrap="CJK"),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontName="NotoKR", fontSize=6.6,
                                 leading=8.5, textColor=INK, wordWrap="CJK"),
        "table_white": ParagraphStyle("table_white", parent=base["BodyText"], fontName="NotoKR",
                                       fontSize=6.5, leading=8.5, textColor=WHITE, wordWrap="CJK"),
        "table_tiny": ParagraphStyle("table_tiny", parent=base["BodyText"], fontName="NotoKR",
                                      fontSize=7, leading=9, textColor=INK, wordWrap="CJK"),
        "table_tiny_white": ParagraphStyle("table_tiny_white", parent=base["BodyText"], fontName="NotoKR",
                                            fontSize=6.8, leading=9, textColor=WHITE, wordWrap="CJK"),
        "card_label": ParagraphStyle("card_label", parent=base["BodyText"], fontName="NotoKR",
                                     fontSize=7, leading=9, textColor=MUTED, alignment=TA_CENTER),
        "card_value": ParagraphStyle("card_value", parent=base["BodyText"], fontName="NotoKR",
                                     fontSize=14, leading=18, textColor=NAVY, alignment=TA_CENTER),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName="NotoKR",
                                   fontSize=8.2, leading=12, textColor=INK, wordWrap="CJK"),
    }


def P(value, style):
    return Paragraph(str(value), style)


def band_table(rows, widths, tiny=False):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BLUE_PALE]),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 if tiny else 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 if tiny else 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4 if tiny else 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 if tiny else 5),
    ]))
    return table


def draw_cover(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 185, width, 185, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#A9D9D4"))
    canvas.setFont("NotoKR", 8)
    canvas.drawString(20 * mm, height - 26 * mm, "H100 KERNEL RESEARCH PORTFOLIO")
    canvas.setFillColor(WHITE)
    canvas.setFont("NotoKR", 21)
    canvas.drawString(20 * mm, height - 39 * mm, "RMSNorm 연구 누적 기록")
    canvas.setFont("NotoKR", 9)
    canvas.drawString(20 * mm, height - 48 * mm, f"32,768행 BF16 RMSNorm · 실측 기록 {doc.audit_time}")
    canvas.setStrokeColor(colors.HexColor("#44617C"))
    canvas.line(20 * mm, height - 55 * mm, width - 20 * mm, height - 55 * mm)
    canvas.setFillColor(colors.HexColor("#DDEAF2"))
    canvas.setFont("NotoKR", 7)
    canvas.drawString(20 * mm, height - 61 * mm, "저장소: hyscodebase/h100-rmsnorm-lab · PR #67 · Issue #66")
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "2~18쪽은 2026-11-27 22:24 KST의 연구 기록입니다. 이번 폭별 실측은 19쪽부터 이어집니다.")
    canvas.restoreState()


def draw_appendix(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(20 * mm, height - 14 * mm, width - 20 * mm, height - 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("NotoKR", 7)
    canvas.drawString(20 * mm, height - 11 * mm, "H100 RMSNorm · 폭별 커널 비교")
    canvas.drawRightString(width - 20 * mm, 10 * mm, f"{doc.page + 17}쪽 · {doc.audit_time}")
    canvas.restoreState()


def latest_status(audit, pin, raw_records):
    expected = audit["expected_checkpoints"]
    completed = audit["successful_checkpoints"]
    seconds = sum(sum(float(p.get("event_seconds", 0)) for p in r.get("sustained_phases", []))
                  for r in raw_records)
    samples = [s for r in raw_records for phase in r.get("sustained_phases", [])
               for s in phase.get("telemetry_samples", [])]
    eligible = sum(bool(p.get("activity_telemetry_eligible"))
                   for r in raw_records for p in r.get("sustained_phases", []))
    failed = [r for r in raw_records if r.get("status") != "complete"]
    return {
        "completed": completed, "expected": expected, "seconds": seconds,
        "samples": len(samples), "eligible": eligible,
        "phase_count": sum(len(r.get("sustained_phases", [])) for r in raw_records),
        "scope": pin["h100_execution_scope"], "revision": pin["source_revision"],
        "failed": failed,
    }


def next_unit(plan, audited):
    failed = next((item for item in audited["checkpoint_results"]
                   if item.get("status") != "complete"), None)
    if failed:
        return f"{failed.get('checkpoint', 'unknown')} (blocked)"
    done = {(item.get("seed"), item.get("width")) for item in audited["checkpoint_results"]
            if item.get("status") == "complete"}
    for seed in plan["seeds"]:
        for width in plan["widths"]:
            if (seed, width) not in done:
                return f"s{seed}-w{width}"
    return "complete"


def read_active_data():
    audit = json.loads(AUDIT_PATH.read_text())
    pin = json.loads(PIN_PATH.read_text())
    plan = json.loads((ROOT / "plans/width-dispatch.json").read_text())
    raw_records = []
    for item in audit["checkpoint_results"]:
        path = ROOT / item["raw_path"] if "raw_path" in item else CHECKPOINT_DIR / (item["checkpoint"] + ".json")
        raw_bytes = path.read_bytes()
        record = json.loads(raw_bytes)
        receipt_name = f"{path.stem}-{item['attempt_id']}.json"
        receipt = json.loads((path.parent.parent / "attempts" / receipt_name).read_text())
        if hashlib.sha256(raw_bytes).hexdigest() != receipt.get("checkpoint_sha256"):
            raise SystemExit(f"Checkpoint differs from its receipt: {path.stem}.")
        if (record["attempt_id"] != item["attempt_id"] or record["status"] != item["status"]
                or record["source_revision"] != item.get("source_revision", audit["source_revision"])):
            raise SystemExit(f"Checkpoint changed since the saved audit: {item['checkpoint']}.")
        raw_records.append(record)
    if (audit.get("plan") != "pass" or audit.get("source_pin") != "pass"
            or audit.get("source_revision") != pin.get("source_revision")):
        raise SystemExit("Refusing to publish a PDF from a failed plan or source-pin audit.")
    if len(raw_records) != audit.get("saved_checkpoints"):
        raise SystemExit("Raw checkpoint count differs from the current partial audit.")
    return audit, pin, plan, raw_records


def median_or_none(values):
    present = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.median(present) if present else None


def build_appendix(path, audit, pin, plan, raw_records):
    s = styles()
    status = latest_status(audit, pin, raw_records)
    comparisons = {f"s{x['seed']}-w{x['width']}": x for x in audit["checkpoint_comparisons"]}
    next_name = next_unit(plan, audit)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=17 * mm,
                            title="H100 RMSNorm Research Record - current active data")
    latest = max(raw["finished_utc"] for raw in raw_records)
    doc.audit_time = datetime.fromisoformat(latest).astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    story = [Spacer(1, 154), P("현재 실험 상태", s["section"])]

    card_rows = [
        [P("측정 완료 / 계획", s["card_label"]), P("GPU 지속 측정 시간", s["card_label"])],
        [P(f"{status['completed']} / {status['expected']}", s["card_value"]),
         P(f"{status['seconds'] / 60:.2f}분", s["card_value"])],
        [P("자원 사용량 표본", s["card_label"]), P("사용률 표본 기준을 통과한 구간", s["card_label"])],
        [P(f"{status['samples']:,}행", s["card_value"]),
         P(f"{status['eligible']} / {status['phase_count']} 구간", s["card_value"])],
    ]
    cards = Table(card_rows, colWidths=[84 * mm, 84 * mm], rowHeights=[18, 28, 18, 28])
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE_PALE),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [cards, Spacer(1, 10)]

    if status["failed"]:
        failed = status["failed"][0]
        failure = failed.get("failure", {})
        failed_name = f"s{failed.get('seed')}-w{failed.get('width')}"
        failure_text = (f"{failed_name}의 실행이 {escape(str(failure.get('stage', 'unknown')))} 단계에서 중단됐습니다. "
                        f"{escape(str(failure.get('message', '')))} "
                        f"지연 측정 {len(failed.get('latency_intervals', [])):,}건과 지속 구간 "
                        f"{len(failed.get('sustained_phases', []))}개가 저장됐습니다.")
        alert = Table([[P(failure_text, s["callout"])]], colWidths=[168 * mm])
        alert.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), RED_PALE),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D78B88")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story += [alert, Spacer(1, 8)]

    summary_text = ("4행 Triton 커널이 가장 빠른 1행 커널과 torch.compile 양쪽보다 처리량을 5% 이상 높이는 폭을 찾고 있습니다. "
                    "한 폭을 채택하려면 세 시드가 모두 이 기준을 통과해야 합니다. ")
    if comparisons:
        last = list(comparisons.values())[-1]
        gains = last["gain_percent"]
        summary_text += (f"최근 등록 순서의 완료 항목은 시드 {last['seed']}, 폭 {last['width']:,}입니다. "
                         f"4행 커널의 지속 처리량 차이는 가장 빠른 1행 대비 {gains['fastest_one_row']:+.2f}%, "
                         f"torch.compile 대비 {gains['torch_compile']:+.2f}%였습니다.")
    if audit.get("final_complete"):
        supported = audit["bounded_dispatch_screen_decision"]["supports_registered_odd_even_split"]
        summary_text += (" 여덟 폭에서 등록한 홀짝 선택 조건이 재현됐습니다." if supported
                         else " 여덟 폭의 결과는 등록한 홀짝 선택 조건에 미달했습니다.")
    summary = Table([[P(summary_text, s["callout"])]], colWidths=[168 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GOLD_PALE),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#E5C17D")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [summary, Spacer(1, 9)]
    if audit.get("predecessor"):
        story += [P("초기 두 측정은 기존 커밋에서 원자료와 영수증을 다시 확인했습니다. 폭 4,107의 초기 드라이버 조회 실패는 "
                    "측정 전에 발생해 당시 지연 표본과 지속 구간이 모두 0개였습니다. 재개 버전은 조회 경로와 오류를 함께 기록합니다.", s["body"])]
    widths = [[P(x, s["table_white"]) for x in ("폭", "완료 시드", "기준 통과 시드", "1행 대비 중앙값", "compile 대비 중앙값")]]
    for width, row in audit["per_width_across_seeds"].items():
        values = [f"{int(width):,}", f"{row['seed_count']} / {len(plan['seeds'])}", str(row["qualifying_seeds"]),
                  f"{row['median_gain_percent_vs_fastest_one_row']:+.2f}%", f"{row['median_gain_percent_vs_torch_compile']:+.2f}%"]
        widths.append([P(value, s["table"]) for value in values])
    if len(widths) > 1:
        story += [P("폭별 지속 처리량 비교", s["subsection"]), band_table(widths, [25*mm, 28*mm, 35*mm, 39*mm, 41*mm], tiny=True)]
    story += [PageBreak(), P("1. 시드·폭별 측정 결과", s["section"])]
    story += [P("latency 중앙값은 us/op, 지속 처리량은 논리 입력+출력 GB/s입니다. 각 네 수치는 4w/1r · 8w/1r · 4w/4r · torch.compile 순입니다.", s["small"])]

    result_rows = [[P(x, s["table_tiny_white"]) for x in (
        "체크포인트", "Latency us/op<br/>4w/1r · 8w/1r · 4w/4r · compile",
        "Sustained GB/s<br/>같은 순서", "후보 vs 1행", "5% gate")]]
    for result in audit["checkpoint_results"]:
        name = result.get("checkpoint", "unknown")
        comp = comparisons.get(name, {})
        if result.get("status") != "complete":
            failure = result.get("failure", {})
            result_rows.append([
                P(f"{name}<br/>실패: {failure.get('stage', 'unknown')}", s["table_tiny"]),
                P("비교 제외", s["table_tiny"]), P("비교 제외", s["table_tiny"]),
                P("n/a", s["table_tiny"]), P("blocked", s["table_tiny"]),
            ])
            continue
        latencies = result.get("latency_median_us_per_op", {})
        phase_rates = {k: v.get("logical_input_plus_output_GB_per_s")
                       for k, v in result.get("phases", {}).items()}
        latency_text = " | ".join(f"{latencies.get(key, float('nan')):.2f}" for key, _ in BACKENDS)
        rate_text = " | ".join(f"{phase_rates.get(key, float('nan')):.1f}" for key, _ in BACKENDS)
        gain = comp.get("gain_percent", {}).get("fastest_one_row")
        result_rows.append([
            P(name, s["table_tiny"]), P(latency_text, s["table_tiny"]), P(rate_text, s["table_tiny"]),
            P("n/a" if gain is None else f"{gain:+.2f}%", s["table_tiny"]),
            P("yes" if comp.get("qualifies") else "no", s["table_tiny"]),
        ])
    story += [band_table(result_rows, [23 * mm, 55 * mm, 55 * mm, 19 * mm, 16 * mm], tiny=True), Spacer(1, 6)]
    story += [P("한 폭을 채택하려면 세 시드 모두에서 가장 빠른 1행 Triton과 torch.compile 각각에 대해 지속 처리량이 5% 이상 높아야 합니다.", s["small"])]
    story += [PageBreak()]

    story += [P("2. 자원 계측과 다음 실행", s["section"]), P(
        "각 sustained phase에서 원시 nvidia-smi 표본과 query timestamp 및 수신 시간을 저장합니다. 1.5초를 넘는 공백이 있으면 해당 phase를 "
        "활동 계측 부적격으로 표시하고 계산 결과와 원시 표본은 보존합니다.", s["body"])]
    telemetry_rows = [[P(x, s["table_tiny_white"]) for x in (
        "체크포인트", "event 분", "표본 행", "busy 중간/최고", "VRAM peak MiB",
        "power 중간/최고 W", "temp peak C", "max gap 초", "eligible")]]
    for raw in raw_records:
        name = f"s{raw['seed']}-w{raw['width']}"
        phases = raw.get("sustained_phases", [])
        samples = [sample for phase in phases for sample in phase.get("telemetry_samples", [])]
        if not phases:
            telemetry_rows.append([
                P(f"{name}<br/>gate failed", s["table_tiny"]), P("0.00", s["table_tiny"]),
                P("0", s["table_tiny"]), P("n/a", s["table_tiny"]), P("n/a", s["table_tiny"]),
                P("n/a", s["table_tiny"]), P("n/a", s["table_tiny"]), P("n/a", s["table_tiny"]),
                P("n/a", s["table_tiny"]),
            ])
            continue
        busy = [x.get("gpu_busy_percent") for x in samples]
        vram = [x.get("memory_used_mib") for x in samples]
        power = [x.get("power_w") for x in samples]
        temp = [x.get("temperature_c") for x in samples]
        max_gap = max((phase.get("telemetry_max_gap_seconds", 0) or 0 for phase in phases), default=0)
        eligible = sum(bool(phase.get("activity_telemetry_eligible")) for phase in phases)
        event_min = sum(float(phase.get("event_seconds", 0) or 0) for phase in phases) / 60
        telemetry_rows.append([
            P(name, s["table_tiny"]), P(f"{event_min:.2f}", s["table_tiny"]),
            P(str(len(samples)), s["table_tiny"]),
            P(f"{median_or_none(busy):.0f}/{max(v for v in busy if v is not None):.0f}" if any(v is not None for v in busy) else "n/a", s["table_tiny"]),
            P(f"{max(v for v in vram if v is not None):,.0f}" if any(v is not None for v in vram) else "n/a", s["table_tiny"]),
            P(f"{median_or_none(power):.1f}/{max(v for v in power if v is not None):.1f}" if any(v is not None for v in power) else "n/a", s["table_tiny"]),
            P(f"{max(v for v in temp if v is not None):.0f}" if any(v is not None for v in temp) else "n/a", s["table_tiny"]),
            P(f"{max_gap:.3f}", s["table_tiny"]), P(f"{eligible}/{len(phases)}", s["table_tiny"]),
        ])
    story += [band_table(telemetry_rows, [20 * mm, 15 * mm, 16 * mm, 25 * mm, 19 * mm,
                                         25 * mm, 16 * mm, 18 * mm, 14 * mm], tiny=True), Spacer(1, 8)]

    overall_samples = [sample for raw in raw_records for phase in raw.get("sustained_phases", [])
                       for sample in phase.get("telemetry_samples", [])]
    all_busy = [sample["gpu_busy_percent"] for sample in overall_samples if sample.get("gpu_busy_percent") is not None]
    all_power = [sample["power_w"] for sample in overall_samples if sample.get("power_w") is not None]
    all_vram = [sample["memory_used_mib"] for sample in overall_samples if sample.get("memory_used_mib") is not None]
    all_temp = [sample["temperature_c"] for sample in overall_samples if sample.get("temperature_c") is not None]
    gap = max((phase.get("telemetry_max_gap_seconds", 0) or 0
               for raw in raw_records for phase in raw.get("sustained_phases", [])), default=0)
    note = (f"누적 표본 {len(overall_samples):,}행: GPU busy 중간/최고 {median_or_none(all_busy):.1f}/{max(all_busy):.1f}%, "
            f"VRAM peak {max(all_vram):,.0f} MiB, 전력 중간/피크 {median_or_none(all_power):.1f}/{max(all_power):.1f} W, "
            f"온도 peak {max(all_temp):.0f} C. 최대 timestamp gap {gap:.3f}초로 현재 기준 1.5초를 초과한 구간은 정식 활동 사용률 통계로 인용하지 않습니다.")
    box = Table([[P(note, s["callout"])]], colWidths=[168 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED_PALE),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#E5B0AE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [box, Spacer(1, 8)]
    previous = audit.get("predecessor", {})
    lineage = f"처음 두 건의 소스 {previous['source_revision'][:12]}. " if previous else ""
    story += [P(f"{lineage}현재 측정 소스 {pin['source_revision'][:12]}. 실행 범위 {status['scope']}. "
                f"다음 등록 단위 {next_name}. 원자료와 시도 기록은 저장소의 results/width-dispatch/raw/에 있습니다. "
                "초기 리뷰에서 제외한 9회의 측정은 2~18쪽의 이력에 남아 있습니다.", s["small"])]

    doc.build(story, onFirstPage=draw_cover, onLaterPages=draw_appendix)


def main():
    audit, pin, plan, records = read_active_data()
    with tempfile.TemporaryDirectory(prefix="h100-research-pdf-") as temp_dir:
        temp = Path(temp_dir)
        baseline_result = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{BASELINE_PDF}"],
            cwd=ROOT, capture_output=True, check=True,
        )
        appendix_path = temp / "appendix.pdf"
        build_appendix(appendix_path, audit, pin, plan, records)
        baseline = PdfReader(io.BytesIO(baseline_result.stdout))
        appendix = PdfReader(str(appendix_path))
        writer = PdfWriter()
        writer.add_page(appendix.pages[0])
        for page in baseline.pages:
            writer.add_page(page)
        for page in appendix.pages[1:]:
            writer.add_page(page)
        writer.add_metadata({
            "/Title": "H100 RMSNorm 연구 누적 기록 - 최신 활성 데이터",
            "/Author": "hyscodebase",
            "/Subject": "H100 kernel experiments, audit evidence, and active width-dispatch matrix",
        })
        with tempfile.NamedTemporaryFile(dir=OUTPUT_PDF.parent, suffix=".tmp", delete=False) as stream:
            writer.write(stream)
        Path(stream.name).replace(OUTPUT_PDF)
        stats = latest_status(audit, pin, records)
        print(f"updated={OUTPUT_PDF} pages={len(writer.pages)} active={stats['completed']}/{stats['expected']} "
              f"cuda_event_minutes={stats['seconds'] / 60:.2f} telemetry_rows={stats['samples']} "
              f"eligible_phases={stats['eligible']}/{stats['phase_count']}")


if __name__ == "__main__":
    main()
