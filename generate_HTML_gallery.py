#!/usr/bin/env python3
"""
generate_gallery.py
Generates a self-contained HTML figure gallery from a folder of PNG/SVG plots.
Tailored for: auditory-pRF-subcortical dipc_test figures
Filename format: sequence{N}_fc{X}hz_dur{D}ms_isi{I}ms_total{T}sec_numtones{K}_CF_{F}Hz

Usage:
    python generate_gallery.py
    python generate_gallery.py --folder /path/to/other/figures
    python generate_gallery.py --out my_gallery.html
"""

import os
import re
import base64
import argparse
from pathlib import Path

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_FOLDER = "/home/ekim/auditory-pRF-subcortical/figures/dipc_test_250225_01"
DEFAULT_OUT    = "gallery.html"
IMG_EXTS       = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}

# ── filename parser ──────────────────────────────────────────────────────────
PATTERN = re.compile(
    r"sequence(\d+)_fc(\d+)hz_.*_CF_(\d+)Hz",
    re.IGNORECASE
)

def parse_filename(stem: str) -> dict:
    m = PATTERN.search(stem)
    if m:
        return {
            "seq":    int(m.group(1)),
            "fc_hz":  int(m.group(2)),
            "cf_hz":  int(m.group(3)),
        }
    return {"seq": 0, "fc_hz": 0, "cf_hz": 0}

# ── image → base64 ───────────────────────────────────────────────────────────
MIME = {".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}

def img_to_b64(path: Path) -> str:
    ext = path.suffix.lower()
    mime = MIME.get(ext, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

# ── collect & sort files ─────────────────────────────────────────────────────
def collect(folder: Path) -> list[dict]:
    files = [p for p in folder.iterdir()
             if p.suffix.lower() in IMG_EXTS]
    items = []
    for p in files:
        meta = parse_filename(p.stem)
        meta["path"]  = p
        meta["name"]  = p.stem
        items.append(meta)
    # sort by seq first, then CF
    items.sort(key=lambda x: (x["seq"], x["cf_hz"]))
    return items

# ── unique sorted values ─────────────────────────────────────────────────────
def unique(items, key):
    return sorted(set(x[key] for x in items))

# ── HTML generation ───────────────────────────────────────────────────────────
def build_html(items: list[dict], folder_name: str) -> str:
    seqs   = unique(items, "seq")
    fc_vals = unique(items, "fc_hz")
    cf_vals = unique(items, "cf_hz")

    # Build JS data array (embed images as base64 for portability)
    print(f"  Encoding {len(items)} images — this may take a moment...")
    js_items = []
    for it in items:
        try:
            src = img_to_b64(it["path"])
        except Exception:
            src = ""
        js_items.append(
            f'{{seq:{it["seq"]},fc:{it["fc_hz"]},cf:{it["cf_hz"]},'
            f'name:{repr(it["name"])},src:{repr(src)}}}'
        )
    js_data = "[\n    " + ",\n    ".join(js_items) + "\n  ]"

    seq_opts  = "\n".join(f'<option value="{v}">{v}</option>' for v in seqs)
    fc_opts   = "\n".join(f'<option value="{v}">{v} Hz</option>' for v in fc_vals)
    cf_opts   = "\n".join(f'<option value="{v}">{v} Hz</option>' for v in cf_vals)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Figure Gallery — {folder_name}</title>
<style>
  :root {{
    --bg:      #0e0f14;
    --surface: #16181f;
    --border:  #2a2d3a;
    --accent:  #5b8dee;
    --accent2: #e05b8d;
    --text:    #d4d8f0;
    --muted:   #6b7094;
    --radius:  8px;
    --font:    'IBM Plex Mono', 'Courier New', monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    min-height: 100vh;
  }}

  /* ── top bar ── */
  header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  header h1 {{
    font-size: 14px;
    letter-spacing: .08em;
    color: var(--accent);
    flex: 0 0 auto;
    margin-right: 8px;
  }}
  .tag {{
    background: #1e2030;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    color: var(--muted);
    font-size: 11px;
  }}
  .spacer {{ flex: 1; }}
  #count {{
    color: var(--muted);
    font-size: 11px;
  }}

  /* ── filter bar ── */
  #filters {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .filter-group {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .filter-group label {{
    color: var(--muted);
    font-size: 11px;
    letter-spacing: .06em;
    white-space: nowrap;
  }}
  select, input[type=range] {{
    background: #1e2030;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    padding: 4px 8px;
    font-family: var(--font);
    font-size: 12px;
    cursor: pointer;
  }}
  select:focus {{ outline: 1px solid var(--accent); }}
  #search {{
    background: #1e2030;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    padding: 5px 10px;
    font-family: var(--font);
    font-size: 12px;
    width: 220px;
  }}
  #search::placeholder {{ color: var(--muted); }}
  button {{
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--muted);
    padding: 4px 12px;
    font-family: var(--font);
    font-size: 11px;
    cursor: pointer;
    transition: color .15s, border-color .15s;
  }}
  button:hover {{ color: var(--text); border-color: var(--accent); }}

  /* ── grid ── */
  #grid {{
    padding: 20px 24px;
    display: grid;
    grid-template-columns: repeat(var(--cols, 5), 1fr);
    gap: 10px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    cursor: pointer;
    transition: border-color .15s, transform .1s;
    position: relative;
  }}
  .card:hover {{
    border-color: var(--accent);
    transform: translateY(-2px);
  }}
  .card img {{
    width: 100%;
    display: block;
    aspect-ratio: 1.4;
    object-fit: contain;
    background: #0a0b10;
  }}
  .card-label {{
    padding: 5px 7px;
    font-size: 10px;
    color: var(--muted);
    line-height: 1.5;
    border-top: 1px solid var(--border);
  }}
  .card-label strong {{ color: var(--accent); }}

  /* ── lightbox ── */
  #lightbox {{
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,.88);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    gap: 12px;
  }}
  #lightbox.open {{ display: flex; }}
  #lb-img {{
    max-width: 92vw;
    max-height: 82vh;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #0a0b10;
    object-fit: contain;
  }}
  #lb-info {{
    color: var(--text);
    font-size: 12px;
    letter-spacing: .05em;
    text-align: center;
  }}
  #lb-close {{
    position: fixed;
    top: 18px; right: 24px;
    font-size: 22px;
    background: none;
    border: none;
    color: var(--muted);
    cursor: pointer;
    line-height: 1;
  }}
  #lb-close:hover {{ color: var(--text); }}
  #lb-nav {{
    display: flex;
    gap: 16px;
  }}
  #lb-nav button {{
    padding: 6px 20px;
    font-size: 13px;
  }}

  /* ── cols slider label ── */
  #cols-val {{
    color: var(--accent);
    min-width: 16px;
    display: inline-block;
    text-align: center;
  }}

  /* ── empty state ── */
  #empty {{
    display: none;
    padding: 60px 24px;
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    grid-column: 1/-1;
  }}
</style>
</head>
<body>

<header>
  <h1>▸ FIGURE GALLERY</h1>
  <span class="tag">{folder_name}</span>
  <span class="tag">{len(items)} figures · 40 seq × 40 CF</span>
  <div class="spacer"></div>
  <span id="count"></span>
</header>

<div id="filters">
  <div class="filter-group">
    <label>SEARCH</label>
    <input id="search" type="text" placeholder="filename fragment…">
  </div>
  <div class="filter-group">
    <label>SEQUENCE</label>
    <select id="f-seq"><option value="">ALL</option>{seq_opts}</select>
  </div>
  <div class="filter-group">
    <label>FC (carrier)</label>
    <select id="f-fc"><option value="">ALL</option>{fc_opts}</select>
  </div>
  <div class="filter-group">
    <label>CF (channel)</label>
    <select id="f-cf"><option value="">ALL</option>{cf_opts}</select>
  </div>
  <div class="filter-group">
    <label>COLS <span id="cols-val">5</span></label>
    <input type="range" id="cols-slider" min="1" max="10" value="5" style="width:90px">
  </div>
  <button id="btn-reset">RESET</button>
</div>

<div id="grid"><div id="empty">No figures match the current filters.</div></div>

<!-- lightbox -->
<div id="lightbox">
  <button id="lb-close">✕</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-info"></div>
  <div id="lb-nav">
    <button id="lb-prev">◀ PREV</button>
    <button id="lb-next">NEXT ▶</button>
  </div>
</div>

<script>
const DATA = {js_data};

let filtered = [...DATA];
let lbIdx = 0;

const grid      = document.getElementById('grid');
const empty     = document.getElementById('empty');
const countEl   = document.getElementById('count');
const lightbox  = document.getElementById('lightbox');
const lbImg     = document.getElementById('lb-img');
const lbInfo    = document.getElementById('lb-info');
const colsSlider = document.getElementById('cols-slider');
const colsVal   = document.getElementById('cols-val');

function render() {{
  // remove old cards
  Array.from(grid.querySelectorAll('.card')).forEach(c => c.remove());

  if (filtered.length === 0) {{
    empty.style.display = 'block';
    countEl.textContent = '0 shown';
    return;
  }}
  empty.style.display = 'none';
  countEl.textContent = filtered.length + ' shown';

  filtered.forEach((item, i) => {{
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <img src="${{item.src}}" alt="${{item.name}}" loading="lazy">
      <div class="card-label">
        <strong>SEQ ${{item.seq}}</strong> · fc ${{item.fc}} Hz · CF ${{item.cf}} Hz
      </div>`;
    card.addEventListener('click', () => openLightbox(i));
    grid.appendChild(card);
  }});
}}

function applyFilters() {{
  const q   = document.getElementById('search').value.toLowerCase();
  const seq = document.getElementById('f-seq').value;
  const fc  = document.getElementById('f-fc').value;
  const cf  = document.getElementById('f-cf').value;

  filtered = DATA.filter(d => {{
    if (q   && !d.name.toLowerCase().includes(q))      return false;
    if (seq && d.seq  !== parseInt(seq))               return false;
    if (fc  && d.fc   !== parseInt(fc))                return false;
    if (cf  && d.cf   !== parseInt(cf))                return false;
    return true;
  }});
  render();
}}

// filter listeners
['search','f-seq','f-fc','f-cf'].forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters));

document.getElementById('btn-reset').addEventListener('click', () => {{
  document.getElementById('search').value = '';
  document.getElementById('f-seq').value  = '';
  document.getElementById('f-fc').value   = '';
  document.getElementById('f-cf').value   = '';
  applyFilters();
}});

colsSlider.addEventListener('input', () => {{
  const v = colsSlider.value;
  colsVal.textContent = v;
  grid.style.setProperty('--cols', v);
}});

// lightbox
function openLightbox(i) {{
  lbIdx = i;
  lightbox.classList.add('open');
  showLb();
}}

function showLb() {{
  const it = filtered[lbIdx];
  lbImg.src = it.src;
  lbInfo.textContent = it.name + '  |  seq ' + it.seq + '  ·  fc ' + it.fc + ' Hz  ·  CF ' + it.cf + ' Hz';
}}

document.getElementById('lb-close').addEventListener('click', () =>
  lightbox.classList.remove('open'));
document.getElementById('lb-prev').addEventListener('click', () => {{
  lbIdx = (lbIdx - 1 + filtered.length) % filtered.length; showLb();
}});
document.getElementById('lb-next').addEventListener('click', () => {{
  lbIdx = (lbIdx + 1) % filtered.length; showLb();
}});
lightbox.addEventListener('click', e => {{
  if (e.target === lightbox) lightbox.classList.remove('open');
}});
document.addEventListener('keydown', e => {{
  if (!lightbox.classList.contains('open')) return;
  if (e.key === 'ArrowRight') {{ lbIdx = (lbIdx+1) % filtered.length; showLb(); }}
  if (e.key === 'ArrowLeft')  {{ lbIdx = (lbIdx-1+filtered.length) % filtered.length; showLb(); }}
  if (e.key === 'Escape')     lightbox.classList.remove('open');
}});

// init
render();
</script>
</body>
</html>"""
    return html

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate HTML figure gallery")
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--out",    default=DEFAULT_OUT)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"ERROR: folder not found: {folder}")
        return

    print(f"Scanning: {folder}")
    items = collect(folder)
    print(f"  Found {len(items)} figures")

    if not items:
        print("No image files found. Check the folder path and extensions.")
        return

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = folder.parent / out_path

    html = build_html(items, folder.name)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Gallery written → {out_path}")
    print(f"  Open in browser: firefox '{out_path}' &")

if __name__ == "__main__":
    main()