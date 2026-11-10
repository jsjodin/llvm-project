#!/bin/sh
# Renders the Markdown documents in this directory to PDF, beside them:
# pandoc makes HTML with a table of contents, and headless Chrome prints it.
#
#   render-pdf.sh             every .md file here
#   render-pdf.sh FILE.md...  only those
#
# Needs pandoc and google-chrome (or chromium, or $CHROME). The PDFs are
# ignored by git (merkurius/.gitignore).

set -eu

DOCS=$(dirname "$(readlink -f "$0")")

if [ -n "${CHROME:-}" ]; then
  :
elif command -v google-chrome >/dev/null 2>&1; then
  CHROME=google-chrome
elif command -v chromium >/dev/null 2>&1; then
  CHROME=chromium
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROME=chromium-browser
else
  echo "render-pdf.sh: needs google-chrome or chromium, or CHROME set" >&2
  exit 1
fi
command -v pandoc >/dev/null 2>&1 || {
  echo "render-pdf.sh: needs pandoc" >&2
  exit 1
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/style.css" <<'EOF'
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "DejaVu Serif", "Noto Serif", serif; font-size: 10.5pt;
  line-height: 1.45; color: #111; max-width: 100%; }
h1, h2, h3, h4 { font-family: "DejaVu Sans", "Noto Sans", sans-serif;
  line-height: 1.2; page-break-after: avoid; }
h1 { font-size: 20pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt;
  margin-top: 22pt; }
h3 { font-size: 12.5pt; margin-top: 16pt; }
h4 { font-size: 11pt; margin-top: 12pt; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt;
  background: #f3f3f3; padding: 0 2px; border-radius: 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6pt 8pt;
  font-size: 8.5pt; white-space: pre-wrap; word-wrap: break-word;
  page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt;
  page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; vertical-align: top; }
th { background: #eee; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 10pt;
  color: #444; }
nav#TOC { page-break-after: always; font-family: "DejaVu Sans", sans-serif;
  font-size: 9.5pt; }
nav#TOC ul { list-style: none; padding-left: 14pt; }
nav#TOC > ul { padding-left: 0; }
a { color: #114; text-decoration: none; }
EOF

# pandoc before 2.19 has --self-contained, later ones --embed-resources.
if pandoc --help | grep -q -- --embed-resources; then
  EMBED=--embed-resources
else
  EMBED=--self-contained
fi

if [ $# -eq 0 ]; then
  set -- "$DOCS"/*.md
fi

for md in "$@"; do
  name=$(basename "$md" .md)
  dir=$(dirname "$(readlink -f "$md")")
  pandoc "$md" -f gfm -t html5 --standalone --toc --toc-depth=3 $EMBED \
    --css "$WORK/style.css" --metadata title="$name" -o "$WORK/$name.html"
  timeout 300 "$CHROME" --headless=new --disable-gpu --no-sandbox \
    --user-data-dir="$WORK/chrome" --no-pdf-header-footer \
    --print-to-pdf="$dir/$name.pdf" "$WORK/$name.html" >/dev/null 2>&1
  echo "$dir/$name.pdf"
done
