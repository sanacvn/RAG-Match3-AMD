#!/usr/bin/env python3
"""Generate independent dev/holdout corpora and labels OUTSIDE each corpus."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_split(dest, seed):
    import pymupdf
    from docx import Document
    from openpyxl import Workbook
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    corpus = dest / "corpus"
    for sub in ("empty", "notes", "tables", "drawings", "archive", "events", "source", "private"):
        (corpus / sub).mkdir(parents=True, exist_ok=True)
    labels = []
    def label(query, answer, citations, category):
        labels.append({"query_id": f"synthetic_{len(labels)+1:02d}", "query": query,
                       "expected_answer": answer, "expected_citations": citations,
                       "answerable": bool(answer), "category": category})
    product = f"PX-{rng.randint(200,999)}"
    # Ten simple questions exercise CPU diagnostics without implying LLM quality.
    for i in range(10):
        field = f"{product} register {i}"
        value = f"V-{rng.randint(10000,99999)}"
        path = f"notes/record-{i}.txt"
        (corpus/path).write_text(f"{field}: {value}\n")
        label(f"What is the {field}?", value, [path], "key_value")
    quarter = f"Q{rng.randint(1,4)} FY{rng.randint(29,39)}"
    doc = Document()
    doc.add_heading("Approved roadmap", 0)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Product", "Customer sampling quarter"
    cells = table.add_row().cells
    cells[0].text, cells[1].text = product, quarter
    doc.save(corpus / "tables/plan.docx")
    label(f"When does {product} enter customer sampling?", quarter, ["tables/plan.docx"], "docx_table")
    part = f"FAN-{rng.randint(3000,9000)}-R"
    book = Workbook()
    book.active.title = "Unrelated"
    book.active.append(["Description", "Value"])
    book.active.append(["Other part", "X-0"])
    sheet = book.create_sheet("Service parts")
    sheet.append(["Product", "Field replaceable fan part number"])
    sheet.append([product, part])
    book.save(corpus / "tables/parts.xlsx")
    label(f"What is the field replaceable fan part number for {product}?", part, ["tables/parts.xlsx"], "xlsx_second_sheet")
    ticket, version = f"BUG-{rng.randint(1000,9000)}", f"{rng.randint(5,9)}.2.{rng.randint(1,9)}"
    (corpus / "events/run.log").write_text(f"2029-02-02 {product} thermal throttle triggered; ticket {ticket}\n")
    with (corpus / "tables/fixes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ticket", "Fixed firmware version"])
        w.writerow([ticket, version])
    label(f"Which firmware version fixed ticket {ticket}?", version, ["tables/fixes.csv"], "csv")
    label(f"Which firmware fixed the thermal throttle defect in the production log for {product}?", version,
          ["events/run.log", "tables/fixes.csv"], "cross_file")
    timeout = str(rng.randint(150,300))
    (corpus / "source/service.py").write_text(f"# {product} batch timeout in seconds\nBATCH_TIMEOUT = {timeout}\nraise RuntimeError('CORPUS MUST NEVER EXECUTE')\n")
    label(f"What is the default batch timeout in seconds for {product}?", timeout, ["source/service.py"], "python")
    temperature = str(rng.randint(80,100))
    for name, status, temp in [("archive/historic.pdf", "WITHDRAWN; superseded by current specification", "71"),
                               ("notes/current.pdf", "ACTIVE approved specification", temperature)]:
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((40,60), f"{product}\nStatus: {status}\nMaximum junction temperature: {temp} C")
        pdf.save(corpus/name)
        pdf.close()
    label(f"What is the current maximum junction temperature of {product}?", temperature, ["notes/current.pdf"], "revision")
    label(f"What was the maximum junction temperature in the withdrawn specification for {product}?", "71", ["archive/historic.pdf"], "historical_revision")
    rev = f"REV-D{rng.randint(1,9)}"
    pin = f"C{rng.randint(10,40)}"
    for name, text in [("drawings/board.jpg", f"Board {product} | Revision: {rev}"),
                       ("drawings/pins.png", f"{product} | THERM_ALERT# pin: {pin}")]:
        image = Image.new("RGB", (1000,200), "white")
        ImageDraw.Draw(image).text((20,60), text, fill="black", font_size=36)
        image.save(corpus/name)
    label("What board revision is printed on the asset photograph?", rev, ["drawings/board.jpg"], "image")
    label(f"Which pin carries THERM_ALERT# for {product}?", pin, ["drawings/pins.png"], "image")
    label(f"What is the unit price of {product} at 20000 units?", "", [], "unanswerable")
    pdf = pymupdf.open()
    pdf.new_page().insert_text((40,60), "Confidential unit price: 777")
    pdf.save(corpus / "private/locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256,
             owner_pw="synthetic-owner", user_pw="synthetic-password")
    pdf.close()
    (corpus / "private/unknown.dat").write_bytes(b"unparsed")
    denied = corpus / "private/unreadable.txt"
    denied.write_text("Must not be read when permissions disallow access")
    denied.chmod(0)
    (corpus / "private/broken.docx").write_text("not a zip")
    (dest / "labels.json").write_text(json.dumps(labels, indent=2)+"\n")
    return len(labels)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "work/synthetic")
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output directory to preserve existing labels/splits")
    # The initial 20261019 split exposed a diagnostic CSV value-format bug and
    # was promoted to development. This is a fresh, untuned holdout seed.
    for split, seed in [("dev", 20261011), ("holdout", 20261023)]:
        print(split, make_split(args.output/split, seed))
