"""Local browser GUI for labeling a blinded prepared-remark audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .contracts import ALLOWED_LABELS

# The historical model-comparison runs retain their four-label contract.
AUDIT_LABELS = ALLOWED_LABELS | {"general_business_context"}
USEFULNESS_LABELS = frozenset({"yes", "no"})

LABEL_BUTTONS = (
    ("1", "boilerplate", "Boilerplate"),
    ("2", "repeated_substantive", "Repeated substantive"),
    ("3", "novel_substantive", "Novel substantive"),
    ("4", "administrative", "Administrative"),
    ("5", "general_business_context", "General business context"),
)
assert frozenset(label for _, label, _ in LABEL_BUTTONS) == AUDIT_LABELS


def load_audit(path: Path) -> tuple[dict, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Audit file does not exist: {path}")
    records = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audit_id = record.get("audit_id")
            if audit_id is None:
                raise ValueError(f"Missing audit_id at {path}:{line_number}")
            if audit_id in seen_ids:
                raise ValueError(f"Duplicate audit_id {audit_id!r} in {path}")
            seen_ids.add(audit_id)
            records.append(record)
    if not records:
        raise ValueError(f"Audit file is empty: {path}")
    return tuple(records)


def load_label_events(path: Path, *, allowed_labels=AUDIT_LABELS) -> tuple[dict, ...]:
    if not path.exists():
        return ()
    events = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if "audit_id" not in event or "label" not in event:
                raise ValueError(f"Invalid label event at {path}:{line_number}")
            label = event["label"]
            if label is not None and label not in allowed_labels:
                raise ValueError(f"Invalid label {label!r} at {path}:{line_number}")
            events.append(event)
    return tuple(events)


def current_labels(events: tuple[dict, ...]) -> dict[int, str]:
    labels = {}
    for event in events:
        audit_id = int(event["audit_id"])
        if event["label"] is None:
            labels.pop(audit_id, None)
        else:
            labels[audit_id] = event["label"]
    return labels


def append_label_event(
    path: Path,
    *,
    audit_id: int,
    sentence_id: str,
    label: str | None,
    allowed_labels=AUDIT_LABELS,
) -> None:
    if label is not None and label not in allowed_labels:
        raise ValueError(f"Invalid audit label: {label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "audit_id": audit_id,
        "sentence_id": sentence_id,
        "label": label,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False) + "\n")
        output.flush()


def next_unlabeled(audit: tuple[dict, ...], labels: dict[int, str]):
    return next(
        (record for record in audit if int(record["audit_id"]) not in labels),
        None,
    )


def _page(*, reviewed: int, total: int, item: dict | None, message: str = "", usefulness=False) -> bytes:
    percentage = 100.0 if total == 0 else reviewed / total * 100
    if item is None:
        body = """
        <main class="complete">
          <h1>Audit complete</h1>
          <p>Every sentence has been labeled. You can undo the last label below.</p>
        </main>
        """
    else:
        controls = (("1", "yes", "Useful for generation"), ("2", "no", "Not useful")) if usefulness else LABEL_BUTTONS
        buttons = "".join(
            f'<button class="label label-{key}" name="label" value="{label}">'
            f'<span class="key">{key}</span>{escape(title)}</button>'
            for key, label, title in controls
        )
        cross_match = item.get("cross_company_match")
        cross_panel = ""
        if cross_match:
            cross_panel = f"""
              <section>
                <div class="section-label">Closest different-company call</div>
                <p>{escape(cross_match)}</p>
              </section>
            """
        novelty_reference = "both matches" if cross_match else "the previous-call match"
        body = f"""
        <main>
          <form method="post" action="/label">
            <input type="hidden" name="audit_id" value="{int(item['audit_id'])}">
            <section class="current">
              <div class="section-label">Sentence to label</div>
              <p>{escape(item['current_sentence'])}</p>
            </section>
            <div class="comparison-grid">
              <section>
                <div class="section-label">Closest previous company call</div>
                <p>{escape(item['previous_call_match'])}</p>
              </section>
              {cross_panel}
            </div>
            <div class="buttons">{buttons}</div>
          </form>
          <details>
            <summary>Label definitions</summary>
            <dl>
              <dt>Boilerplate</dt><dd>Generic reusable disclosure with little period-specific substance.</dd>
              <dt>Repeated substantive</dt><dd>Repeated topic or wording that still makes a meaningful company claim.</dd>
              <dt>Novel substantive</dt><dd>A meaningful company claim that is materially different from {novelty_reference}.</dd>
              <dt>Administrative</dt><dd>Introductions, speaker transitions, call instructions, or document references.</dd>
              <dt>General business context</dt><dd>Recurring explanations, broad commentary, or general outlook without a specific result, event, change, or actionable claim.</dd>
            </dl>
          </details>
        </main>
        """
        if usefulness:
            body = f"""
            <main><form method="post" action="/label">
              <input type="hidden" name="audit_id" value="{int(item['audit_id'])}">
              <section class="current">
                <div class="section-label">Could this remark guide a specific, worthwhile financial question?</div>
                <p>{escape(item['current_sentence'])}</p>
              </section>
              <div class="buttons">{buttons}</div>
            </form>
            <p>Yes — a specific result, event, development, or comparison with enough context to identify the claim.</p>
            <p>No — vague, generic, incomplete, or dependent on missing context. A filing match is assessed later.</p>
            </main>
            """
    message_html = f'<div class="message">{escape(message)}</div>' if message else ""
    document = f"""<!doctype html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Boilerplate sentence audit</title>
    <style>
      :root {{ color-scheme: light; --ink:#17202a; --muted:#617080; --line:#d8dee5;
              --panel:#f7f9fb; --accent:#2557a7; }}
      * {{ box-sizing:border-box; }}
      body {{ font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; color:var(--ink);
              max-width:1100px; margin:0 auto; padding:24px; line-height:1.55; background:#fff; }}
      header {{ display:flex; justify-content:space-between; gap:20px; align-items:end;
                margin-bottom:18px; }}
      h1 {{ font-size:1.35rem; margin:0; }} .progress-text {{ color:var(--muted); }}
      .bar {{ height:8px; background:#e8edf2; border-radius:8px; overflow:hidden; margin-bottom:24px; }}
      .bar span {{ display:block; height:100%; width:{percentage:.2f}%; background:var(--accent); }}
      section {{ border:1px solid var(--line); border-radius:10px; padding:20px; background:var(--panel); }}
      section p {{ font-size:1.05rem; margin:.55rem 0 0; }}
      .current {{ border:2px solid #8ba9d6; background:#f8fbff; }}
      .current p {{ font-size:1.2rem; }}
      .section-label {{ color:var(--muted); font-size:.78rem; font-weight:700;
                        text-transform:uppercase; letter-spacing:.06em; }}
      .comparison-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0; }}
      .buttons {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:20px 0; }}
      button {{ border:1px solid #aeb9c5; border-radius:8px; background:white; padding:13px 10px;
                font:inherit; font-weight:650; cursor:pointer; }}
      button:hover {{ border-color:var(--accent); background:#f2f6fc; }}
      .key {{ display:inline-grid; place-items:center; width:24px; height:24px; margin-right:8px;
              border-radius:5px; color:white; background:var(--accent); }}
      .footer {{ display:flex; justify-content:space-between; align-items:center; margin-top:18px; }}
      .undo {{ color:var(--muted); padding:8px 12px; font-weight:500; }}
      .message {{ color:#17612d; }} details {{ color:var(--muted); margin-top:14px; }}
      dt {{ font-weight:700; color:var(--ink); margin-top:8px; }} dd {{ margin-left:0; }}
      @media(max-width:760px) {{ .comparison-grid,.buttons {{ grid-template-columns:1fr; }} }}
    </style></head><body>
      <header><h1>{"Generation-usefulness audit" if usefulness else "Prepared-remark similarity audit"}</h1>
        <div class="progress-text">Reviewed {reviewed} of {total}</div></header>
      <div class="bar"><span></span></div>
      {message_html}{body}
      <div class="footer">
        <span>{"Keys 1–2 rate usefulness." if usefulness else "Keys 1–5 label the current sentence."}</span>
        <form method="post" action="/undo"><button class="undo">[u] Undo last label</button></form>
      </div>
      <script>document.addEventListener('keydown', event => {{
        if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return;
        const button = [...document.querySelectorAll('button[name=label]')]
          .find(candidate => candidate.querySelector('.key')?.textContent === event.key);
        if (button) {{ event.preventDefault(); button.click(); }}
        if (event.key.toLowerCase() === 'u') {{
          event.preventDefault(); document.querySelector('form[action="/undo"] button').click();
        }}
      }});</script>
    </body></html>"""
    return document.encode("utf-8")


def make_handler(audit_path: Path, labels_path: Path, *, usefulness=False):
    audit = load_audit(audit_path)
    audit_by_id = {int(record["audit_id"]): record for record in audit}
    allowed = USEFULNESS_LABELS if usefulness else AUDIT_LABELS

    class AuditHandler(BaseHTTPRequestHandler):
        def _redirect(self, message: str = "") -> None:
            suffix = f"?message={quote(message)}" if message else ""
            self.send_response(303)
            self.send_header("Location", "/" + suffix)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/":
                self.send_error(404)
                return
            labels = current_labels(load_label_events(labels_path, allowed_labels=allowed))
            message = parse_qs(parsed.query).get("message", [""])[0]
            page = _page(
                reviewed=len(labels),
                total=len(audit),
                item=next_unlabeled(audit, labels),
                message=message,
                usefulness=usefulness,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            data = parse_qs(self.rfile.read(length).decode("utf-8"))
            events = load_label_events(labels_path, allowed_labels=allowed)
            labels = current_labels(events)

            if path == "/label":
                try:
                    audit_id = int(data.get("audit_id", [""])[0])
                except ValueError:
                    self.send_error(400, "Invalid audit ID.")
                    return
                label = data.get("label", [""])[0]
                item = audit_by_id.get(audit_id)
                if item is None or label not in allowed:
                    self.send_error(400, "Invalid audit item or label.")
                    return
                if audit_id in labels:
                    self._redirect("Item was already labeled")
                    return
                append_label_event(
                    labels_path,
                    audit_id=audit_id,
                    sentence_id=item["sentence_id"],
                    label=label,
                    allowed_labels=allowed,
                )
                self._redirect("Saved")
                return

            if path == "/undo":
                active_id = None
                for event in reversed(events):
                    audit_id = int(event["audit_id"])
                    if event["label"] is not None and audit_id in labels:
                        active_id = audit_id
                        break
                if active_id is None:
                    self._redirect("Nothing to undo")
                    return
                item = audit_by_id[active_id]
                append_label_event(
                    labels_path,
                    audit_id=active_id,
                    sentence_id=item["sentence_id"],
                    label=None,
                    allowed_labels=allowed,
                )
                self._redirect("Undid last label")
                return

            self.send_error(404)

        def log_message(self, format, *args) -> None:  # noqa: A003
            return

    return AuditHandler


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--task", choices=("content", "usefulness"), default="content")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    audit_path = args.audit.resolve()
    labels_path = (
        args.labels.resolve()
        if args.labels
        else audit_path.with_name(
            f"{audit_path.stem}_{'usefulness_' if args.task == 'usefulness' else ''}labels.jsonl"
        )
    )
    if labels_path == audit_path:
        raise ValueError("The label ledger must be separate from the blinded audit file.")
    audit = load_audit(audit_path)
    usefulness = args.task == "usefulness"
    allowed = USEFULNESS_LABELS if usefulness else AUDIT_LABELS
    labels = current_labels(load_label_events(labels_path, allowed_labels=allowed))
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(audit_path, labels_path, usefulness=usefulness)
    )
    print(f"Review at http://127.0.0.1:{args.port}/")
    print(f"Audit: {audit_path}")
    print(f"Labels: {labels_path}")
    print(f"Progress: {len(labels)} of {len(audit)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
