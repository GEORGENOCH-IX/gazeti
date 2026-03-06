"""
pipeline/headline_parser.py
Detects headlines and section labels from raw OCR text.
"""

import re


class HeadlineParser:
    """
    Scans raw OCR text line by line and groups content into
    articles by detecting section headers and headlines.

    Headlines in Nation papers tend to be:
      - Title Case or ALL CAPS
      - Between 4 and 18 words long
      - Preceded by a section label (National News, Health, etc.)
    """

    SECTION_PATTERN = re.compile(
        r"(National News|Health|Business|Sports|"
        r"Opinion|Politics|Counties|World)",
        re.IGNORECASE,
    )

    def parse(self, raw_text: str) -> list[dict]:
        """
        Splits OCR text into a list of article dicts, each with:
          - section:    e.g. "National News"
          - headline:   e.g. "MP, 4 Others Honoured in Emotional Mass"
          - body_lines: list of paragraph lines below the headline
        """
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        articles = []
        current = {"section": "", "headline": "", "body_lines": []}

        for line in lines:

            # ── Detect section header ──────────────────────────────
            if self.SECTION_PATTERN.search(line) and len(line) < 40:
                current["section"] = line
                continue

            # ── Detect headline ────────────────────────────────────
            word_count = len(line.split())
            is_title_case = line.istitle() and 4 <= word_count <= 18
            is_all_caps   = line.isupper() and 4 <= word_count <= 18

            if is_title_case or is_all_caps:
                # Save the previous article before starting a new one
                if current["headline"] and current["body_lines"]:
                    articles.append(dict(current))

                current = {
                    "section":    current["section"],  # carry section forward
                    "headline":   line,
                    "body_lines": [],
                }

            else:
                # Everything else is body text
                current["body_lines"].append(line)

        # Don't forget the last article on the page
        if current["headline"]:
            articles.append(current)

        return articles