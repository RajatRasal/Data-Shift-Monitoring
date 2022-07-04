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
