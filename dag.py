from typing import List

from dagster import get_dagster_logger, graph, op, repository
from dagster_prometheus.resources import prometheus_resource
from prometheus_client import Gauge

from models.data import (
    PDFPageImage,
    get_images_from_pdf,
    find_pdfs_by_glob,
)
from models.ocr import PDFOCRResult
from resources import file_systems, search_indices, models


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


@op(required_resource_keys={"data_drift_model"})
def monitor_data_drift(context, input_pdfs: List[PDFPageImage]) -> float:
    # TODO: Considerations about data lineage
    # TODO: Better wrapping for img_arr and predict function
    # TODO: Store image embedding from inside the model - histogram
    # TODO: Store embeddings from PCA - histogram
    # TODO: Data drift model should be specific to the true distirbution of
    #    a particular production model.
    img_arr = [pdf.image for pdf in input_pdfs]
    # TODO: If this is a large batch split up, return List[float]
    drift_score = context.resources.data_drift_model.predict(img_arr)
    get_dagster_logger().info(f"Drift score: {drift_score}")
    return drift_score


@op(required_resource_keys={"prometheus"})
def store_data_drift(context, data_drift_score: float):
    # Model id, dagster run id, score
    time_gauge = Gauge(
        'drift_score_time_unixtime',
        'Time when drift score was calculated',
        registry=context.resources.prometheus.registry,
    )
    drift_score_gauge = Gauge(
        'drift_score_mse',
        'Drift score',
        registry=context.resources.prometheus.registry,
    )
    time_gauge.set_to_current_time()
    drift_score_gauge.set(data_drift_score)
    context.resources.prometheus.push_to_gateway(
        job='drift_score',
        grouping_key={
            "ocr_model_version": "test",
            "drift_model_version": "test",
        },
    )


# TODO: Dynamic out for each image K8s executor. 
# (config_schema={"tracking_uri": str, "version": int})
@op(required_resource_keys={"ocr_model"})
def ocr_predictions(
    context,
    datapoints: List[PDFPageImage],
) -> List[PDFOCRResult]:
    ocr_results = []
    for i, data in enumerate(datapoints):
        if i % 100 == 0:
            get_dagster_logger().info(f"Processing image {i + 1}")
        # TODO: Maybe do this batchwise using toolz + DynamicOut
        results = context.resources.ocr_model.predict(data)
        ocr_results.append(PDFOCRResult(data, results))
    return ocr_results


@op
def calculate_prediction_metrics(results: List[PDFOCRResult]):
    # TODO: Number of words detected
    # TODO: Size of boxes detected
    # Per image metrics
    for pdf_result in results:
        words_count = 0
        size = []
        for detection in pdf_result.detections:
            words_count += len(detection.text.split(" "))
            size = []
    pass


@op(required_resource_keys={"search_index"})
def store_prediction_metrics(context, result: List[PDFOCRResult]):
    pass


@op(required_resource_keys={"fs", "search_index"})
def store_predictions(context, result: List[PDFOCRResult]):
    # store writing to es and images to s3
    pass


def _pipeline(pdf_paths):
    datapoints = pdfs_to_images(pdf_paths)
    store_data_drift(monitor_data_drift(datapoints))
    # predictions = ocr_predictions(datapoints)
    # store_prediction_metrics(calculate_prediction_metrics(predictions))
    # store_predictions(predictions)


@graph
def pipeline():
    pdf_paths = find_pdfs()
    _pipeline(pdf_paths)


@graph
def failure_pipeline():
    pdf_paths = reprocess_failures()
    _pipeline(pdf_paths)


@repository
def repo():
    # TODO: Automatic trigger with minio when new data is dumped into it.
    job = pipeline.to_job(
        name="OCR_test_job",
        resource_defs={
            "fs": file_systems.local_file_system,
            "ocr_model": models.ocr_model,
            "data_drift_model": models.data_drift_model,
            "reconstruction_model": models.reconstruction_model,
            "search_index": search_indices.elasticsearch,
            # TODO: Use prometheus for data logging.
            "prometheus": prometheus_resource,
        },
        config={"ops": {"find_pdfs": {"config": {"input_path": "OCR_TEXT"}}},
        "resources": {"prometheus": {"config": {"gateway": "localhost:9091"}}}},
    )
    return [job]
