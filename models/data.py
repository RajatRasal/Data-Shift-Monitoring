import hashlib
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Generator, Union, Tuple

import numpy as np
from pdf2image import convert_from_path


COORD_TYPE = Tuple[int, int]

@dataclass
class OCRResult:
    boxes: Tuple[COORD_TYPE, COORD_TYPE, COORD_TYPE, COORD_TYPE]
    text: str
    confidence: float


@dataclass
class PDFPageImage:
    pdf_name: str
    page_no: int
    image: np.ndarray


@dataclass
class PDFOCRResult:
    pdf_page_image: PDFPageImage
    detections: List[OCRResult]


def find_pdfs_by_glob(fs, dir: str) -> List[str]:
    return fs.glob(f"{dir}/**pdf")


def get_images_from_pdf(
    fs,
    remote_file_name: str,
    local_file_name: str
) -> List[PDFPageImage]:
    try:
        fs.get(remote_file_name, local_file_name)
    except Exception as e:
        # TODO: File downloading error
        # Get stack trace
        raise DownloadException(
            msg=f"Cannot download {remote_file_name}",
            further_info=str(e)
        )

    try:
        images = convert_from_path(local_file_name)
    except Exception as e:
        # TODO: File loading error
        raise ImageToPDFConversionError(
            msg=f"Cannot convert {local_file_name} to pdf",
            further_info=str(e)
        )

    return [
        PDFPageImage(
            remote_file_name,
            page_no,
            np.array(image).astype(np.uint8),
        )
        for page_no, image in enumerate(images)
    ]


def create_es_actions(
    results: List[PDFOCRResult],
    run_id: str,
    ocr_model_version: int,
    index: str,
) -> Generator[Dict[str, Union[str, int, float]], None, None]:
    # TODO: Change to a generator to reduce memory consumption
    for result in results:
        for ocr_result_id, det in enumerate(result.detections):
            m = hashlib.sha256()
            m.update(result.pdf_page_image.pdf_name.encode('utf-8'))
            es_record = {
                "dagster_run_id": run_id,
                "ocr_model_version": ocr_model_version,
                "pdf_name_hash": m.hexdigest(),
                "pdf_name": result.pdf_page_image.pdf_name,
                "page_no": result.pdf_page_image.page_no,
                "ocr_result_id": ocr_result_id,
                "boxes": det.boxes,
                "text": det.text,
                "confidence": det.confidence,
            }
            _id = "-".join([
                str(es_record['ocr_model_version']),
                es_record['pdf_name_hash'],
                str(es_record['page_no']),
                str(es_record['ocr_result_id']),
            ])
            action = {
                "_index": index,
                # TODO: Include doc_type here
                # "_type": doc_type,
                "_id": _id,
                "_source": es_record,
            }
            yield action
