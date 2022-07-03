import tempfile
from typing import List

from PIL import Image
from dagster import get_dagster_logger, graph, op, resource
from fsspec.implementations.local import LocalFileSystem

import easyocr
from pdf2image import convert_from_path


@dataclass
class PDFPageImage:
    pdf_name: str
    page_no: int
    image: Image


@resource
def file_system():
    return LocalFileSystem()


@resource
def es_writer():
    return None


@op(config_schema={"input_path": str}, required_resource_keys={"fs"})
def find_pdfs(context) -> List[str]:
    pdf_glob = f"{context.op_config['input_path']}/**pdf"
    return context.resources.fs.glob(pdf_glob)


@op(required_resource_keys={"fs"})
def pdfs_to_images(context, pdfs: List[str]) -> List[PDFPageImage]:
    results = []
    if pdfs:
        for pdf in pdfs:
            # TODO: Logging each pdf completion
            _tempfile = tempfile.NamedTemporaryFile()
            context.resources.fs.get(pdf, _tempfile.name)
            images = convert_from_path(_tempfile.name)
            for page_no, image in enumerate(images):
                results.append(PDFPageImage(pdf, page_no, image))
            _tempfile.close()
    get_dagster_logger().info(f"Found {len(results)} images")
    return results


@op(required_resource_keys={"data_logger"})
def monitor_data_drift(input_pdfs: List[PDFPageImage]):
    # TODO: Considerations about data lineage
    pass


@op  # (config_schema={"tracking_uri": str, "version": int})
def ocr_predictions(context, datapoints: List[PDFPageImage]) -> List[PDFPageImage]:
    model = easyocr.Reader(
        lang_list=["en"],
        download_enabled=True,
        # model_storage_directory=models_dir,
    )

    for data in datapoints:
        # TODO: Logging every N image completion
        # TODO: Maybe do this batchwise
        res = model.readtext(
            image=data.image,
            output_format="dict",
        )
        data.text = res

    return datapoints


@op(required_resource_keys={"search_index"})
def store_prediction_metrics(context, result: List[PDFPageImage]):
    pass

@op(required_resource_keys={"fs", "search_index"})
def store_predictions(context, result: List[PDFPageImage]):
    # store writing to s3 and images to f3
    pass


@graph
def pipeline():
    datapoints = pdfs_to_images(find_pdfs())
    monitor_data_drift(datapoints)
    predictions = ocr_predictions(datapoints)
    store_prediction_metrics(predictions)
    store_predictions(predictions)


if __name__ == "__main__":
    job = pipeline.to_job(
        name="OCR_test_job",
        resource_defs={
            "fs": file_system,
            "search_index": es_writer,
            "data_logger": es_writer,
        },
    )
    job.execute_in_process()
