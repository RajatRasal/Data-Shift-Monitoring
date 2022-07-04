from dagster import resource

from models.ocr import OCRModel
from models.data_drift import PCADriftModel


@resource
def ocr_model():
    return OCRModel()


@resource
def data_drift_model():
    return PCADriftModel()