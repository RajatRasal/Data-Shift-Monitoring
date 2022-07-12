from dagster import resource

from models.data_drift import DRIFT_PICKLE_FILE, DriftModel, PCAPipeline
from models.ocr import OCRModel


@resource
def ocr_model():
    return OCRModel()


@resource(required_resource_keys={"reconstruction_model"})
def data_drift_model(init_context):
    model = init_context.resources.reconstruction_model
    return DriftModel(model)


@resource
def reconstruction_model():
    pca_pipeline = PCAPipeline.load(DRIFT_PICKLE_FILE)
    return pca_pipeline
