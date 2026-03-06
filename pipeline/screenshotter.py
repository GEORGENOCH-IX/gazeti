"""
pipeline/screenshotter.py
Captures e-paper pages as screenshots using a headless browser.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright


class EpaperScreenshotter:
    """
    Navigates the Daily Nation e-paper flip-book
    and saves each page as a PNG screenshot.
    """

    BASE_URL = (
        "https://epaper.nation.africa/read/ip/{issue_id}"
        "?whitelistId=41&brand=EP_DAILY_NATION"
    )

    def __init__(self, issue_id: str = "13861", output_dir: str = "screenshots"):
        self.issue_id = issue_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def capture_pages(self, start_page: int = 1, end_page: int = 5) -> list[Path]:
        """
        Opens the e-paper in a headless browser,
        navigates page by page, saves screenshots.
        Returns a list of saved image paths.
        """
        url = self.BASE_URL.format(issue_id=self.issue_id)
        saved = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 900}
            )
            page = context.new_page()

            print(f"[screenshotter] Loading: {url}")
            page.goto(url, wait_until="networkidle", timeout=60_000)
            time.sleep(3)  # let the flip-book JavaScript initialise

            for page_num in range(start_page, end_page + 1):
                # Navigate forward one page at a time
                if page_num > 1:
                    page.keyboard.press("ArrowRight")
                    time.sleep(1.5)

                path = self.output_dir / f"page_{page_num:03d}.png"
                page.screenshot(path=str(path))
                saved.append(path)
                print(f"  ✓ Page {page_num} saved → {path}")

            browser.close()

        return saved