import tempfile
from typing import List

from dagster import get_dagster_logger, graph, op, resource

from model.data import PDFPageImage, get_images_from_pdf
from model.ocr import load_model
from resources.file_systems import local_file_system
from resources.search_indices import elasticsearch


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
            # TODO: Error handling for get_images_from_pdf - image need not be pdf
            _images = get_images_from_pdf(context.resources.fs, pdf)
            results.extend(_images)
    get_dagster_logger().info(f"Found {len(results)} images")
    return results


@op(required_resource_keys={"data_logger"})
def monitor_data_drift(input_pdfs: List[PDFPageImage]):
    # TODO: Considerations about data lineage
    pass


@op  # (config_schema={"tracking_uri": str, "version": int})
def ocr_predictions(context, datapoints: List[PDFPageImage]) -> List[PDFPageImage]:
    model = load_model()

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
            "fs": local_file_system,
            "search_index": elasticsearch,
            "data_logger": elasticsearch,
        },
    )
    job.execute_in_process()
