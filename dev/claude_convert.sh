#!/usr/bin/env bash
# pdf_pages_to_markdown.sh
#
# Convert directories of PDF page images to Markdown using Claude Code agents.
# One agent per page image. Phase 1 transcribes pages in parallel; phase 2
# concatenates per-PDF outputs.
#
# Layout assumed:
#   INPUT_DIR/
#     doc_a/
#       page_001.png
#       page_002.png
#     doc_b/
#       page_001.png
#       ...
#
# Layout produced:
#   OUTPUT_DIR/
#     doc_a.md                    # final concatenated markdown
#     doc_b.md
#     .pages/doc_a/page_001.md    # per-page intermediates (kept for resumability)
#     .pages/doc_a/page_002.md
#     .logs/doc_a/page_001.log    # per-page agent logs
#
# Usage:
#   ./pdf_pages_to_markdown.sh INPUT_DIR OUTPUT_DIR [PARALLEL]
#
# Env:
#   MODEL    Claude model alias (default: claude-sonnet-4-6)
#
# Requires: claude (Claude Code CLI), bash 3.2+ (works on stock macOS), xargs
# Auth:     run `claude login` once to authenticate via your Pro/Max subscription.
#           If ANTHROPIC_API_KEY is set in your environment, Claude Code uses the
#           API key (and bills the API account) instead of your subscription —
#           unset it first if you want subscription billing.
# Tip:      run from a directory without project-level .claude/ or CLAUDE.md to
#           keep the agent's behaviour clean. User-level ~/.claude/ config still loads.

set -euo pipefail

INPUT_DIR="${1:?usage: $0 INPUT_DIR OUTPUT_DIR [PARALLEL]}"
OUTPUT_DIR="${2:?usage: $0 INPUT_DIR OUTPUT_DIR [PARALLEL]}"
PARALLEL="${3:-8}"

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "warning: ANTHROPIC_API_KEY is set; Claude Code will use it and bill the API account." >&2
    echo "         unset it before running if you want to use your Pro/Max subscription." >&2
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"

PAGES_DIR="$OUTPUT_DIR/.pages"
LOGS_DIR="$OUTPUT_DIR/.logs"
mkdir -p "$PAGES_DIR" "$LOGS_DIR"

MODEL="${MODEL:-claude-opus-4-7}"

read -r -d '' PROMPT_TEMPLATE <<'EOF' || true
You are transcribing ONE page of a PDF (provided as an image) to Markdown.

Image to read:    __IMAGE__
Output file path: __OUTPUT__

Steps:
1. Use Read to view the image at the path above.
2. Transcribe its visible contents to Markdown (rules below).
3. Use Write to save the Markdown to the output file path above.
4. Print a one-line summary, e.g. "Wrote <N> words".

CRITICAL TRANSCRIPTION RULES:
- Transcribe EVERY visible word verbatim. Do NOT summarise. Do NOT paraphrase.
  Do NOT skip paragraphs because they "seem unimportant". If you can read it, write it.
- A page with 500 visible words should produce ~500 words of markdown.
- Read characters carefully. Distinguish similar glyphs (rn vs m, l vs 1, O vs 0,
  cl vs d). If a word is genuinely illegible, write [illegible] — do NOT guess.
- Preserve heading hierarchy with #, ##, ###.
- Render tables as Markdown; fall back to HTML <table> only for merged/nested cells.
  Include every row and column.
- Render math in LaTeX: $inline$ or $$display$$.
- Figures: ![<short alt describing what is visibly in the figure>](figure).
  Do NOT invent content not visible in the figure.
- Drop repeating page headers, footers, and standalone page numbers.
- If text is cut off at the top/bottom (cross-page continuation), transcribe only
  what is visible on this page. Do NOT invent continuations.
- Preserve list structure and fenced code blocks.
- Multi-column layouts: finish the left column before the right.

Output ONLY the markdown body in the file. No preamble, no commentary,
no "Here is the markdown:" intro, no fences around the whole file.
EOF

convert_image() {
    local image_path="$1"
    local pdf_name; pdf_name="$(basename "$(dirname "$image_path")")"
    local image_basename; image_basename="$(basename "$image_path")"
    local image_stem="${image_basename%.*}"

    local page_dir="$PAGES_DIR/$pdf_name"
    local log_dir="$LOGS_DIR/$pdf_name"
    local output="$page_dir/$image_stem.md"
    local logfile="$log_dir/$image_stem.log"

    mkdir -p "$page_dir" "$log_dir"

    if [[ -s "$output" ]]; then
        echo "[skip] $pdf_name/$image_stem"
        return 0
    fi

    local prompt="${PROMPT_TEMPLATE//__IMAGE__/$image_path}"
    prompt="${prompt//__OUTPUT__/$output}"

    echo "[start] $pdf_name/$image_stem"
    if ! claude -p "$prompt" \
        --model "$MODEL" \
        --allowedTools "Read,Write" \
        --max-turns 6 \
        --effort="high" \
        --output-format text \
        > "$logfile" 2>&1
    then
        echo "[fail] $pdf_name/$image_stem (agent error; see $logfile)" >&2
        return 1
    fi

    if [[ ! -s "$output" ]]; then
        echo "[fail] $pdf_name/$image_stem (no output written; see $logfile)" >&2
        return 1
    fi

    echo "[done] $pdf_name/$image_stem"
}

export -f convert_image
export PAGES_DIR LOGS_DIR PROMPT_TEMPLATE MODEL

# --- Phase 1: transcribe every image -----------------------------------------
IMAGES=()
while IFS= read -r f; do
    IMAGES+=("$f")
done < <(find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) \
    | sort)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "no images found in $INPUT_DIR/*/" >&2
    exit 1
fi

echo "phase 1: transcribing ${#IMAGES[@]} pages, $PARALLEL agents in parallel (model=$MODEL)"
printf '%s\n' "${IMAGES[@]}" | \
    xargs -I {} -P "$PARALLEL" bash -c 'convert_image "$1"' _ {} || true
# `|| true`: keep going even if some images failed; phase 2 will report missing pages.

# --- Phase 2: concatenate per-PDF --------------------------------------------
echo "phase 2: concatenating per-PDF outputs"

PDF_DIRS=()
while IFS= read -r d; do
    PDF_DIRS+=("$d")
done < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

for pdf_dir in "${PDF_DIRS[@]}"; do
    pdf_name="$(basename "$pdf_dir")"
    page_dir="$PAGES_DIR/$pdf_name"
    output="$OUTPUT_DIR/$pdf_name.md"

    n_pages=$(find "$pdf_dir" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)

    page_files=()
    if [[ -d "$page_dir" ]]; then
        while IFS= read -r f; do
            page_files+=("$f")
        done < <(find "$page_dir" -maxdepth 1 -type f -name '*.md' | sort)
    fi

    if [[ ${#page_files[@]} -eq 0 ]]; then
        echo "[skip] $pdf_name (no per-page markdown found)" >&2
        continue
    fi

    : > "$output"
    first=1
    for f in "${page_files[@]}"; do
        if (( first )); then first=0; else printf '\n\n' >> "$output"; fi
        cat "$f" >> "$output"
    done

    if [[ ${#page_files[@]} -lt $n_pages ]]; then
        echo "[warn]   $pdf_name -> $output (${#page_files[@]}/$n_pages pages — some failed, check $LOGS_DIR/$pdf_name)"
    else
        echo "[merged] $pdf_name -> $output (${#page_files[@]} pages)"
    fi
done

echo "done."