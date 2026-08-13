"""
PDF page images -> Markdown via the Claude Agent SDK.

One agent per image, parallel processing bounded by --parallel, then per-PDF
concatenation. Composition of small focused classes:

    Workspace            -- filesystem layout (input, output, .pages, .logs)
    PdfDocument          -- one PDF folder; knows its pages and how to concatenate
    PageImage            -- one page image; knows its output and log paths
    PageTranscriber      -- the only class that talks to claude_agent_sdk
    TranscriptionPipeline-- orchestrates phase 1 (transcribe) and phase 2 (concat)

Auth:    run `claude login` once to authenticate with your Pro/Max subscription.
         If ANTHROPIC_API_KEY is set, the SDK uses it (and bills the API account)
         instead of your subscription. Unset it for subscription billing.

Install: pip install claude-agent-sdk

Usage:
    python pdf_pages_to_markdown.py INPUT_DIR OUTPUT_DIR [--parallel N] [--model M]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from enum import Enum
from functools import cached_property
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

# --- prompts -----------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an OCR-quality transcriber for PDF page images.

Your role is to transcribe what is LITERALLY VISIBLE on a page. You do not
summarise. You do not paraphrase. You do not complete thoughts from context.
You do not generate plausible academic prose. You do not invent figure
descriptions, tables, or equations. Your output reflects only what is in the
image.

If you catch yourself about to write content that is not visibly on the page,
STOP. Either mark it [illegible] or omit it. Confabulation is a critical
failure mode.

Output rules:
- Tables: every header and every cell, exact values. Numbers verbatim — no
  rounding (0.0053 is not 0.005). Markdown tables; HTML <table> for merged
  or complex cells.
- Math: LaTeX. $inline$ or $$display$$.
- Headings: #, ##, ### matching the visual hierarchy on the page.
- Figures: ![<short alt describing only what is actually visible>](figure).
  Do NOT invent. If a figure shows a 3x3 grid of histograms, say so. Do not
  guess what the data represents.
- Repeating page headers, footers, and standalone page numbers: drop them.
- Cross-page continuations: transcribe what's visible on THIS page; do NOT
  invent continuations.
- Multi-column layouts: each column top-to-bottom before moving right.

Sparse pages produce sparse output. A long output for a short page is a
confabulation signal. If the page is mostly blank, write little.
"""

PROMPT_TEMPLATE = """\
Transcribe this single page image.

Image to read:    {image}
Write output to:  {output}

Steps:
1. Use Read to view the image.
2. Inventory what is actually on the page (headings, paragraphs, tables,
   figures). Be honest about what you can and cannot read clearly.
3. Transcribe verbatim to Markdown per the system rules.
4. Use Write to save the Markdown to the output path above.
5. Reply "done" and stop. Do not echo the content back.
"""

log = logging.getLogger("pdf2md")


# --- result type -------------------------------------------------------------


class PageStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


# --- filesystem layout -------------------------------------------------------


class Workspace:
    """The on-disk layout for a transcription run."""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()

    @property
    def pages_dir(self) -> Path:
        return self.output_dir / ".pages"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / ".logs"

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def discover_documents(self) -> list[PdfDocument]:
        return [
            PdfDocument(d, self) for d in sorted(self.input_dir.iterdir()) if d.is_dir()
        ]


# --- document and page -------------------------------------------------------


class PdfDocument:
    """One PDF, represented as a directory of page images."""

    def __init__(self, source_dir: Path, workspace: Workspace) -> None:
        self.source_dir = source_dir
        self.workspace = workspace
        self.name = source_dir.name

    @cached_property
    def pages(self) -> list[PageImage]:
        return [
            PageImage(p, self)
            for p in sorted(self.source_dir.iterdir())
            if p.is_file() and p.suffix.lower() in Workspace.IMAGE_EXTS
        ]

    @property
    def page_dir(self) -> Path:
        return self.workspace.pages_dir / self.name

    @property
    def log_dir(self) -> Path:
        return self.workspace.logs_dir / self.name

    @property
    def output_md(self) -> Path:
        return self.workspace.output_dir / f"{self.name}.md"

    def concatenate(self) -> tuple[int, int]:
        """Join all per-page markdown files into the final output.

        Returns (n_written, n_expected).
        """
        n_expected = len(self.pages)
        if not self.page_dir.exists():
            return (0, n_expected)
        page_files = sorted(self.page_dir.glob("*.md"))
        if not page_files:
            return (0, n_expected)
        self.output_md.write_text("\n\n".join(p.read_text() for p in page_files))
        return (len(page_files), n_expected)

    def __repr__(self) -> str:
        return f"<PdfDocument {self.name} ({len(self.pages)} pages)>"


class PageImage:
    """One page image from a PDF document."""

    def __init__(self, path: Path, document: PdfDocument) -> None:
        self.path = path
        self.document = document
        self.stem = path.stem

    @property
    def output_md(self) -> Path:
        return self.document.page_dir / f"{self.stem}.md"

    @property
    def log_file(self) -> Path:
        return self.document.log_dir / f"{self.stem}.log"

    @property
    def is_done(self) -> bool:
        return self.output_md.exists() and self.output_md.stat().st_size > 0

    def ensure_dirs(self) -> None:
        self.document.page_dir.mkdir(parents=True, exist_ok=True)
        self.document.log_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"<PageImage {self.document.name}/{self.stem}>"


# --- the SDK boundary --------------------------------------------------------


class PageTranscriber:
    """Transcribes one PageImage to Markdown via the Claude Agent SDK.

    This is the only class in the module that knows about claude_agent_sdk.
    Swap it out for a different implementation (e.g. direct Anthropic SDK)
    without touching anything else.
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str = SYSTEM_PROMPT,
        prompt_template: str = PROMPT_TEMPLATE,
        max_turns: int = 5,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.max_turns = max_turns

    def _options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=self.system_prompt,
            allowed_tools=["Read", "Write"],
            max_turns=self.max_turns,
            permission_mode="acceptEdits",
            # don't auto-load ~/.claude/ hooks/skills/CLAUDE.md
            setting_sources=[],
        )

    async def transcribe(self, page: PageImage) -> PageStatus:
        if page.is_done:
            log.info("[skip] %s/%s", page.document.name, page.stem)
            return PageStatus.SKIPPED

        page.ensure_dirs()
        prompt = self.prompt_template.format(image=page.path, output=page.output_md)
        log_lines: list[str] = []

        log.info("[start] %s/%s", page.document.name, page.stem)
        try:
            async for message in query(prompt=prompt, options=self._options()):
                self._record(message, log_lines)
        except Exception as e:  # noqa: BLE001 -- we want to capture and continue
            log_lines.append(f"EXCEPTION: {e!r}")
            page.log_file.write_text("\n".join(log_lines))
            log.error("[fail] %s/%s (exception)", page.document.name, page.stem)
            return PageStatus.FAILED

        page.log_file.write_text("\n".join(log_lines))

        if not page.is_done:
            log.error("[fail] %s/%s (no output written)", page.document.name, page.stem)
            return PageStatus.FAILED

        log.info("[done] %s/%s", page.document.name, page.stem)
        return PageStatus.OK

    @staticmethod
    def _record(message: object, log_lines: list[str]) -> None:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    log_lines.append(f"TEXT: {block.text}")
                else:
                    log_lines.append(f"BLOCK: {type(block).__name__} {block!r}")
        elif isinstance(message, ResultMessage):
            log_lines.append(f"RESULT: {getattr(message, 'result', None)!r}")
        else:
            log_lines.append(f"MSG: {type(message).__name__}")


# --- orchestration -----------------------------------------------------------


class TranscriptionPipeline:
    """Coordinate phase 1 (parallel transcription) and phase 2 (concatenation)."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        transcriber: PageTranscriber,
        parallel: int,
    ) -> None:
        self.workspace = workspace
        self.transcriber = transcriber
        self.parallel = parallel

    async def run(self) -> int:
        """Returns a process exit code: 0 ok, 1 nothing to do, 2 some failures."""
        self.workspace.prepare()

        documents = self.workspace.discover_documents()
        if not documents:
            log.error("no PDF subdirectories under %s", self.workspace.input_dir)
            return 1

        all_pages = [page for doc in documents for page in doc.pages]
        if not all_pages:
            log.error("no images found in %s/*/", self.workspace.input_dir)
            return 1

        n_failed = await self._transcribe_all(all_pages)
        self._concatenate_all(documents)

        log.info("done.")
        return 0 if n_failed == 0 else 2

    async def _transcribe_all(self, pages: list[PageImage]) -> int:
        log.info(
            "phase 1: transcribing %d pages, %d agents in parallel (model=%s)",
            len(pages),
            self.parallel,
            self.transcriber.model,
        )
        sem = asyncio.Semaphore(self.parallel)

        async def gated(page: PageImage) -> PageStatus:
            async with sem:
                return await self.transcriber.transcribe(page)

        results = await asyncio.gather(*(gated(p) for p in pages))
        n_ok = results.count(PageStatus.OK)
        n_skipped = results.count(PageStatus.SKIPPED)
        n_failed = results.count(PageStatus.FAILED)
        log.info("phase 1: %d ok, %d skipped, %d failed", n_ok, n_skipped, n_failed)
        return n_failed

    def _concatenate_all(self, documents: list[PdfDocument]) -> None:
        log.info("phase 2: concatenating per-PDF outputs")
        for doc in documents:
            written, expected = doc.concatenate()
            if written == 0:
                log.warning("[skip]   %s (no per-page markdown)", doc.name)
            elif written < expected:
                log.warning(
                    "[warn]   %s -> %s (%d/%d pages, see %s)",
                    doc.name,
                    doc.output_md,
                    written,
                    expected,
                    doc.log_dir,
                )
            else:
                log.info(
                    "[merged] %s -> %s (%d pages)",
                    doc.name,
                    doc.output_md,
                    written,
                )


# --- entry point -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_dir", type=Path, help="dir of PDF subdirs of page images")
    ap.add_argument("output_dir", type=Path, help="dir to write <pdf>.md files")
    ap.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="max concurrent agents (default: 8)",
    )
    ap.add_argument(
        "--model",
        default="opus",
        help="model alias (opus, sonnet, haiku) or full id (default: opus)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        log.warning(
            "ANTHROPIC_API_KEY is set; the SDK will use the API key and bill "
            "the API account. Unset it to use your Pro/Max subscription."
        )

    pipeline = TranscriptionPipeline(
        workspace=Workspace(args.input_dir, args.output_dir),
        transcriber=PageTranscriber(model=args.model),
        parallel=args.parallel,
    )
    sys.exit(asyncio.run(pipeline.run()))


if __name__ == "__main__":
    main()
