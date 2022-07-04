from dataclasses import dataclass
from typing import List, Tuple

import easyocr
import numpy as np
from models.data import PDFPageImage


COORD_TYPE = Tuple[int, int]

@dataclass
class OCRResult:
    boxes: Tuple[COORD_TYPE, COORD_TYPE, COORD_TYPE, COORD_TYPE]
    text: str
    confidence: float


@dataclass
class PDFOCRResult:
    pdf_page_image: PDFPageImage
    detections: List[OCRResult]


def load_model():
    return easyocr.Reader(
        lang_list=["en"],
        download_enabled=True,
        # model_storage_directory=models_dir,
    )


def ocr_predictions(model, image: np.ndarray) -> List[OCRResult]:
    raw_results = model.readtext(image=image, output_format="dict")
    return wrap_raw_results(raw_results)


def wrap_raw_results(raw_results) -> List[OCRResult]:
    return [
        OCRResult(
            tuple([(a, b) for a, b in r["boxes"]]),
            r["text"],
            r["confident"]
        )
        for r in raw_results
    ]
