from typing import List

from dagster import get_dagster_logger, graph, op, repository

from models.data import (
    PDFPageImage,
    get_images_from_pdf,
    find_pdfs_by_glob,
)
from models.ocr import PDFOCRResult, load_model, ocr_predictions
from resources import file_systems
from resources import search_indices


# TODO: Play around with minio instead of local file system.
@op(config_schema={"input_path": str}, required_resource_keys={"fs"})
def find_pdfs(context) -> List[str]:
    input_path = context.op_config['input_path']
    try:
        # TODO: Store state in postgres include resource
        pdf_paths = find_pdfs_by_glob(
            context.resources.fs,
            input_path,
        )
    except Exception as e:
        get_dagster_logger().info(f"Failed trying to find pdfs in {input_path}")
        get_dagster_logger().exception(str(e))
        raise e
    else:
        get_dagster_logger().info(f"Found {len(pdf_paths)} pdfs")
        # TODO: If pdf_paths is empty, do not continue with pipeline.
        return pdf_paths


@op
def reprocess_failures():
    # TODO: Implement
    pass


# TODO: Dynamic out for each pdf image with multiprocess executor. 
@op(required_resource_keys={"fs"})
def pdfs_to_images(context, pdfs: List[str]) -> List[PDFPageImage]:
    results = []
    # TODO: Store each image using distributed file system resources
    #   If the images exist, then get them all, else create them.
    for pdf in pdfs:
        get_dagster_logger().info(f"Converting {pdf}")
        try:
            _images = get_images_from_pdf(context.resources.fs, pdf)
        except Exception as e:
            get_dagster_logger().info(f"Error processing {pdf}")
            get_dagster_logger().exception(str(e))
        else:
            results.extend(_images)
    get_dagster_logger().info(f"Found {len(results)} images")
    # TODO: If results empty do not continue
    return results


@op(required_resource_keys={"data_logger"})
def monitor_data_drift(input_pdfs: List[PDFPageImage]):
    # TODO: Considerations about data lineage
    pass


# TODO: Dynamic out for each image K8s executor. 
@op  # (config_schema={"tracking_uri": str, "version": int})
def ocr_predictions(
    context,
    datapoints: List[PDFPageImage],
) -> List[PDFOCRResult]:
    model = load_model()

    ocr_results = []
    for data in datapoints:
        # TODO: Logging every N image completion
        # TODO: Maybe do this batchwise using toolz + DynamicOut
        results = ocr_predictions(model, data.image)
        ocr_results.append(PDFOCRResult(data, results))

    return ocr_results


@op(required_resource_keys={"search_index"})
def store_prediction_metrics(context, result: List[PDFOCRResult]):
    # TODO: Number of words detected
    # TODO: Size of boxes detected
    pass


@op(required_resource_keys={"fs", "search_index"})
def store_predictions(context, result: List[PDFOCRResult]):
    # store writing to s3 and images to f3
    pass


@graph
def pipeline():
    pdf_paths = find_pdfs()
    datapoints = pdfs_to_images(pdf_paths)
    monitor_data_drift(datapoints)
    predictions = ocr_predictions(datapoints)
    store_prediction_metrics(predictions)
    store_predictions(predictions)


@graph
def failure_pipeline():
    pdf_paths = reprocess_failures()
    datapoints = pdfs_to_images(pdf_paths)
    monitor_data_drift(datapoints)
    predictions = ocr_predictions(datapoints)
    store_prediction_metrics(predictions)
    store_predictions(predictions)


@repository
def repo():
    # TODO: Automatic trigger with minio when new data is dumped into it.
    # TODO: Use grafana for data logging.
    job = pipeline.to_job(
        name="OCR_test_job",
        resource_defs={
            "fs": file_systems.local_file_system,
            "search_index": search_indices.elasticsearch,
            "data_logger": search_indices.elasticsearch,
        },
        config={"ops": {"find_pdfs": {"config": {"input_path": "OCR_TEXT"}}}},
    )
    return [job]
