"""
pipeline/preprocessor.py
Cleans up screenshots before OCR to improve text recognition accuracy.
"""

import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """
    Takes a raw screenshot and applies a series of
    image processing steps to make text cleaner for OCR.
    """

    def preprocess(self, image_path: Path) -> np.ndarray:
        """
        Full preprocessing pipeline:
          1. Load image
          2. Convert to greyscale
          3. Boost contrast (CLAHE)
          4. Binarise (Otsu threshold)
          5. Denoise
        Returns a cleaned numpy array ready for OCR.
        """
        # 1. Load
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # 2. Greyscale — OCR doesn't need colour
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE — enhances local contrast (great for newsprint)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grey = clahe.apply(grey)

        # 4. Otsu binarisation — converts to pure black & white
        _, binary = cv2.threshold(
            grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 5. Denoise — removes specks without blurring text
        denoised = cv2.fastNlMeansDenoising(binary, h=10)

        return denoised