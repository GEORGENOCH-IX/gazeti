"""
main.py
Orchestrates the full Gazeti scraping pipeline.

Usage:
    # Scrape live from the e-paper URL:
    python main.py

    # Or pass a local screenshot for testing:
    python main.py --local path/to/image.png
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict

import pandas as pd

from pipeline import (
    EpaperScreenshotter,
    ImagePreprocessor,
    OCRExtractor,
    HeadlineParser,
    NLPExtractor,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Article:
    headline:      str  = ""
    section:       str  = ""
    body_text:     str  = ""
    keywords:      list = field(default_factory=list)
    locations:     list = field(default_factory=list)
    individuals:   list = field(default_factory=list)
    organisations: list = field(default_factory=list)
    figures:       list = field(default_factory=list)
    page:          int  = 0


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class GazetiPipeline:

    def __init__(self, issue_id: str = "13861", work_dir: str = "pipeline_output"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)

        # Initialise all stages
        self.screenshotter = EpaperScreenshotter(
            issue_id=issue_id,
            output_dir=str(self.work_dir / "screenshots"),
        )
        self.preprocessor  = ImagePreprocessor()
        self.ocr           = OCRExtractor()
        self.parser        = HeadlineParser()
        self.nlp           = NLPExtractor()

    # ── Run from live URL ─────────────────────────────────────────────────────
    def run_from_url(self, start_page: int = 1, end_page: int = 5) -> list[Article]:
        print("\n[pipeline] Capturing screenshots …")
        image_paths = self.screenshotter.capture_pages(start_page, end_page)
        return self._process(image_paths)

    # ── Run from local image (great for testing!) ─────────────────────────────
    def run_from_images(self, image_paths: list[str]) -> list[Article]:
        print("\n[pipeline] Using local images …")
        return self._process([Path(p) for p in image_paths])

    # ── Shared processing ─────────────────────────────────────────────────────
    def _process(self, image_paths: list[Path]) -> list[Article]:
        all_articles = []

        for page_num, img_path in enumerate(image_paths, start=1):
            print(f"\n── Page {page_num}: {img_path.name} ──")

            # Stage 1: clean the image
            clean = self.preprocessor.preprocess(img_path)

            # Stage 2: extract raw text
            raw_text = self.ocr.extract_text(clean)

            # Save OCR output for debugging
            ocr_path = self.work_dir / f"ocr_page_{page_num:03d}.txt"
            ocr_path.write_text(raw_text, encoding="utf-8")
            print(f"  [ocr]    saved → {ocr_path}")

            # Stage 3: detect headlines and sections
            parsed = self.parser.parse(raw_text)
            print(f"  [parser] found {len(parsed)} article(s)")

            # Stage 4: NLP extraction per article
            for p in parsed:
                body   = " ".join(p["body_lines"])
                result = self.nlp.extract(p["headline"] + " " + body)

                all_articles.append(Article(
                    headline      = p["headline"],
                    section       = p["section"],
                    body_text     = body[:2000],
                    keywords      = result["keywords"],
                    locations     = result["locations"],
                    individuals   = result["individuals"],
                    organisations = result["organisations"],
                    figures       = result["figures"],
                    page          = page_num,
                ))

        return all_articles

    # ── Export ────────────────────────────────────────────────────────────────
    def export(self, articles: list[Article]):
        data = [asdict(a) for a in articles]

        # JSON
        json_path = self.work_dir / "articles.json"
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n✓ JSON → {json_path}")

        # CSV (flatten lists to strings)
        rows = []
        for row in data:
            rows.append({
                "headline":      row["headline"],
                "section":       row["section"],
                "page":          row["page"],
                "keywords":      ", ".join(row["keywords"]),
                "locations":     ", ".join(row["locations"]),
                "individuals":   ", ".join(row["individuals"]),
                "organisations": ", ".join(row["organisations"]),
                "figures":       "; ".join(
                    f"{f['value']} ({f['type']})" for f in row["figures"]
                ),
                "body_preview":  row["body_text"][:200],
            })
        csv_path = self.work_dir / "articles.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"✓ CSV  → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gazeti — newspaper scraper")
    parser.add_argument(
        "--local", nargs="+", metavar="IMAGE",
        help="Path(s) to local screenshot(s) instead of scraping live"
    )
    parser.add_argument(
        "--issue", default="13861",
        help="E-paper issue ID (default: 13861)"
    )
    parser.add_argument(
        "--pages", nargs=2, type=int, default=[1, 5],
        metavar=("START", "END"),
        help="Page range to scrape (default: 1 5)"
    )
    args = parser.parse_args()

    pipeline = GazetiPipeline(issue_id=args.issue)

    if args.local:
        articles = pipeline.run_from_images(args.local)
    else:
        articles = pipeline.run_from_url(
            start_page=args.pages[0],
            end_page=args.pages[1],
        )

    pipeline.export(articles)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  TOTAL ARTICLES FOUND: {len(articles)}")
    print(f"{'═'*55}")
    for a in articles:
        print(f"\n📰 [{a.section}] {a.headline}")
        print(f"   👤 {', '.join(a.individuals[:4])  or '—'}")
        print(f"   📍 {', '.join(a.locations[:4])    or '—'}")
        print(f"   🔢 {', '.join(f['value'] for f in a.figures[:4]) or '—'}")
        print(f"   🔑 {', '.join(a.keywords[:4])     or '—'}")


if __name__ == "__main__":
    main()