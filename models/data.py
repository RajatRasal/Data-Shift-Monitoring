import hashlib
import tempfile
from dataclasses import dataclass
from typing import List

import numpy as np
from pdf2image import convert_from_path


@dataclass
class PDFPageImage:
    pdf_name: str
    page_no: int
    image: np.ndarray


def find_pdfs_by_glob(fs, dir: str) -> List[str]:
    return fs.glob(f"{dir}/**pdf")


def get_images_from_pdf(fs, pdf_file_name) -> List[PDFPageImage]:
    results = []
    _tempfile = tempfile.NamedTemporaryFile()
    fs.get(pdf_file_name, _tempfile.name)
    images = convert_from_path(_tempfile.name)
    for page_no, image in enumerate(images):
        img_arr = np.array(image).astype(np.uint8)
        results.append(PDFPageImage(pdf_file_name, page_no, img_arr))
    _tempfile.close()
    return results


def create_es_records(results: List[PDFPageImage], run_id, ocr_model_version):
    # TODO: Change to a generator to reduce memory consumption
    es_records = []
    for result in results:
        for ocr_result_id, det in enumerate(result.detections):
            m = hashlib.sha256()
            m.update(det.pdf_name)
            es_record = {
                "dagster_run_id": run_id,
                "ocr_model_version": ocr_model_version,
                "pdf_name_hash": m.hexdigest(),
                "pdf_name": det.pdf_name,
                "page_no": det.page_no,
                "ocr_result_id": ocr_result_id,
                "boxes": det.boxes,
                "text": det.text,
                "confidence": det.confidence,
            }
            _id = "-".join([
                es_record['ocr_model_version'],
                es_record['pdf_name_hash'],
                es_record['page_no'],
                es_record['ocr_result_id'],
            ])
            es_record["id"] = _id
            es_records.append(es_record)
    return es_records
