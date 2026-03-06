"""
pipeline/ocr_extractor.py
Runs Tesseract OCR on a preprocessed image to extract raw text.
"""

import re
import pytesseract
from PIL import Image
import numpy as np

# Tell pytesseract exactly where Tesseract is installed on your system [uncomment and update the path if needed]
# pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

class OCRExtractor:
    """
    Converts a cleaned image (numpy array) into a string of text
    using Tesseract OCR, configured for newspaper layouts.
    """

    # PSM 3 = fully automatic page segmentation (best for newspapers)
    # OEM 3 = use the best available Tesseract engine
    TESS_CONFIG = "--oem 3 --psm 3"

    def extract_text(self, image_array: np.ndarray) -> str:
        """
        Runs OCR on the image and returns cleaned text.
        """
        pil_image = Image.fromarray(image_array)
        raw_text = pytesseract.image_to_string(
            pil_image, config=self.TESS_CONFIG
        )
        return self._clean(raw_text)

    @staticmethod
    def _clean(text: str) -> str:
        """
        Removes common OCR artefacts and normalises whitespace.
        """
        # Keep only printable ASCII characters
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)

        # Collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)

        # Collapse more than 2 blank lines into just one
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()