from typing import List

import easyocr

from models.data import PDFPageImage, OCRResult


class OCRModel:
    def __init__(self):
        self.model = easyocr.Reader(
            lang_list=["en"],
            download_enabled=True,
            # model_storage_directory=models_dir,
        )
    
    def predict(self, data: PDFPageImage) -> List[OCRResult]:
        raw_results = self.model.readtext(
            image=data.image,
            output_format="dict"
        )
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
