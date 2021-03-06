import os
import tempfile
from typing import List

from PIL import Image
from dagster import (
    get_dagster_logger,
    graph,
    op,
    repository,
    file_relative_path,
    config_from_files, 
)
from dagster_prometheus.resources import prometheus_resource
from prometheus_client import Gauge

from models.data import (
    PDFPageImage,
    PDFOCRResult,
    get_images_from_pdf,
    find_pdfs_by_glob,
    create_es_actions,
)
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
    tmp = tempfile.NamedTemporaryFile()

    for pdf in pdfs:
        get_dagster_logger().info(f"Converting {pdf}")
        try:
            _images = get_images_from_pdf(context.resources.fs, pdf, tmp.name)
        except Exception as e:
            get_dagster_logger().exception(f"Error processing {pdf} + {str(e)}")
        else:
            results.extend(_images)

    tmp.close()

    get_dagster_logger().info(f"Found {len(results)} images")
    # TODO: If results empty do not continue
    return results


@op(required_resource_keys={"data_drift_model"})
def monitor_data_drift(context, input_pdfs: List[PDFPageImage]) -> float:
    # TODO: Better wrapping for img_arr and predict function
    # TODO: Store image embedding from inside the model - histogram
    # TODO: Store embeddings from PCA - histogram
    # TODO: Data drift model should be specific to the true distirbution of
    #    a particular production model.
    # TODO: Batchwise inference in process - maybe using dynamicout if 
    #    the dataset is really big.
    img_arr = [pdf.image for pdf in input_pdfs]
    # TODO: If this is a large batch split up, return List[float]
    drift_score = context.resources.data_drift_model.predict(img_arr)
    get_dagster_logger().info(f"Drift score: {drift_score}")
    return drift_score


@op(required_resource_keys={"prometheus"})
def store_data_drift(context, data_drift_score: float):
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
    # TODO: Include OCR and drift model versions
    context.resources.prometheus.push_to_gateway(
        job='drift_score',
        grouping_key={
            "ocr_model_version": "test",
            "drift_model_version": "test",
            "dagster_run_id": context.run_id,
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


@op(required_resource_keys={"search_index"}, config_schema={"index": str})
def store_text(context, results: List[PDFOCRResult]):
    # TODO: Error handling
    TEST_MODEL_VERSION = 0
    es_records = create_es_actions(
        results,
        context.run_id,
        TEST_MODEL_VERSION,
        context.op_config["index"]
    )
    errors = context.resources.search_index.bulk_write(
        es_records,
        get_dagster_logger(),
    )
    if errors:
        raise Exception(f"Failed to index {len(errors)} documents")


@op(config_schema={"base_dir": str}, required_resource_keys={"fs"})
def store_images(context, page_images: List[PDFPageImage]):
    # Store separate images of each pdf in fs
    # /pdf_name/page_no.img
    # TODO: Error handling - if failed log in postgres
    # TODO: Error handling - if base_dir does not exist
    for page_image in page_images:
        with tempfile.NamedTemporaryFile(mode="wb") as f:
            # Save image to temporary file.
            # TODO: Error handling if we cannot write to the file
            img = Image.fromarray(page_image.image)
            img.save(f, "PNG")

            # Build output folder name.
            fs_dir = os.path.join(
                context.op_config["base_dir"],
                page_image.pdf_name,
            )
            # Create folder if does not exist already.
            # TODO: Error handling
            if not context.resources.fs.isdir(fs_dir):
                context.resources.fs.makedir(fs_dir)
                get_dagster_logger().info(f"Created {fs_dir}")

            fs_filename = os.path.join(
                fs_dir,
                str(page_image.page_no),
            )
            # TODO: Error handling
            if not context.resources.fs.exists(fs_filename):
                context.resources.fs.upload(f.name, fs_filename)
                get_dagster_logger().info(f"Uploaded {fs_filename}")


def _pipeline(pdf_paths):
    datapoints = pdfs_to_images(pdf_paths)
    store_images(datapoints)
    store_data_drift(monitor_data_drift(datapoints))
    predictions = ocr_predictions(datapoints)
    # store_prediction_metrics(calculate_prediction_metrics(predictions))
    store_text(predictions)


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
    local_job = pipeline.to_job(
        name="OCR_local",
        resource_defs={
            "fs": file_systems.local_file_system,
            "ocr_model": models.ocr_model,
            "data_drift_model": models.data_drift_model,
            "reconstruction_model": models.reconstruction_model,
            "search_index": search_indices.elasticsearch,
            # TODO: Use prometheus for data logging.
            "prometheus": prometheus_resource,
        },
        config=config_from_files(
            [file_relative_path(__file__, "config/job_config.yaml")]
        ),
    )
    remote_job = pipeline.to_job(
        name="OCR_remote",
        resource_defs={
            "fs": file_systems.s3_file_system,
            "ocr_model": models.ocr_model,
            "data_drift_model": models.data_drift_model,
            "reconstruction_model": models.reconstruction_model,
            "search_index": search_indices.elasticsearch,
            # TODO: Use prometheus for data logging.
            "prometheus": prometheus_resource,
        },
        config=config_from_files(
            [file_relative_path(__file__, "config/s3_job_config.yaml")]
        ),
    )
    return [local_job, remote_job]
