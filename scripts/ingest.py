"""
Chunk markdown knowledge files, embed them, and write `data/index.pkl`.

Usage (from the repo root):
    python -m scripts.ingest

Chunking strategy:
    - Each H1/H2 section becomes a candidate chunk.
    - If a section exceeds CHUNK_MAX_TOKENS, split further on paragraph boundaries.
    - Each chunk carries metadata: source (filename), section (heading), tags (from filename).

Output:
    A single pickle at settings.index_path containing {chunks, embeddings, model, dim}.
    Commit this file to the repo so Railway can serve it without re-ingesting.
"""

from __future__ import annotations

import re
from pathlib import Path

import tiktoken

from app.config import settings
from app.embeddings import embed_batch
from app.retrieval import save_index

_enc = tiktoken.get_encoding("cl100k_base")

SITE_URL = "https://dannyredel.github.io"

FILE_TO_URL = {
    "profile.md":        f"{SITE_URL}/about.html",
    "experience.md":     f"{SITE_URL}/about.html",
    "education.md":      f"{SITE_URL}/about.html",
    "skills_methods.md": f"{SITE_URL}/about.html",
    "differentiators.md":f"{SITE_URL}/about.html",
    "projects.md":       f"{SITE_URL}/projects.html",
    "research.md":       f"{SITE_URL}/research.html",
    "blog_summaries.md": f"{SITE_URL}/myposts.html",
    "target_roles.md":   f"{SITE_URL}/about.html",
}


def _tok(text: str) -> int:
    return len(_enc.encode(text))


def _split_on_headings(md: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] splitting on H1/H2 lines. Content before the first heading is tagged '_intro'."""
    lines = md.splitlines()
    sections: list[tuple[str, list[str]]] = [("_intro", [])]
    heading_re = re.compile(r"^#{1,2}\s+(.*)")
    for line in lines:
        m = heading_re.match(line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _split_by_paragraph(text: str, target: int, hard_max: int) -> list[str]:
    """Greedy pack paragraphs into chunks that aim for `target` tokens, never exceeding `hard_max`."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for p in paragraphs:
        p_tokens = _tok(p)
        if buf and buf_tokens + p_tokens > hard_max:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = [p], p_tokens
            continue
        buf.append(p)
        buf_tokens += p_tokens
        if buf_tokens >= target:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _tags_from_filename(name: str) -> list[str]:
    return [Path(name).stem.lower()]


def _chunk_file(path: Path) -> list[dict]:
    md = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    for heading, body in _split_on_headings(md):
        full = f"# {heading}\n\n{body}" if heading != "_intro" else body
        if _tok(full) <= settings.chunk_max_tokens:
            pieces = [full]
        else:
            pieces = _split_by_paragraph(body, settings.chunk_target_tokens, settings.chunk_max_tokens)
            pieces = [f"# {heading}\n\n{p}" if heading != "_intro" else p for p in pieces]
        for piece in pieces:
            rows.append(
                {
                    "content": piece,
                    "metadata": {
                        "source": path.name,
                        "section": heading if heading != "_intro" else "",
                        "tags": _tags_from_filename(path.name),
                        "url": FILE_TO_URL.get(path.name, f"{SITE_URL}/about.html"),
                    },
                }
            )
    return rows


def run_ingest() -> int:
    kn_dir = Path(settings.knowledge_dir)
    if not kn_dir.exists():
        raise FileNotFoundError(f"knowledge dir not found: {kn_dir.resolve()}")

    files = sorted(p for p in kn_dir.glob("*.md") if not p.name.startswith("PROPOSAL"))
    print(f"[ingest] indexing {len(files)} files from {kn_dir.resolve()}")

    all_chunks: list[dict] = []
    for f in files:
        rows = _chunk_file(f)
        print(f"[ingest]   {f.name}: {len(rows)} chunks")
        all_chunks.extend(rows)

    if not all_chunks:
        print("[ingest] no chunks produced, exiting")
        return 0

    print(f"[ingest] embedding {len(all_chunks)} chunks in batches of 64")
    all_embeddings: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        all_embeddings.extend(embed_batch([r["content"] for r in batch]))

    out = save_index(all_chunks, all_embeddings)
    print(f"[ingest] wrote {len(all_chunks)} chunks to {out.resolve()}")
    return len(all_chunks)


if __name__ == "__main__":
    run_ingest()
