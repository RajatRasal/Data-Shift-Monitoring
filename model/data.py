import tempfile
from typing import List

from pdf2image import convert_from_path
from PIL import Image


@dataclass
class PDFPageImage:
    pdf_name: str
    page_no: int
    image: Image


def get_images_from_pdf(fs, pdf_file_name) -> List[PDFPageImage]:
    results = []
    _tempfile = tempfile.NamedTemporaryFile()
    fs.get(pdf_file_name, _tempfile.name)
    images = convert_from_path(_tempfile.name)
    for page_no, image in enumerate(images):
        results.append(PDFPageImage(pdf_file_name, page_no, image))
    _tempfile.close()