import argparse
from abc import ABC, abstractmethod
from typing import List

import joblib
import numpy as np
import toolz
from PIL import Image
from fsspec.implementations.local import LocalFileSystem
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize


DRIFT_PICKLE_FILE = "/Users/work/Documents/Data-Shift-Monitoring/data_drift_model.pickle"
RESIZE_DIM = (220, 220)


class LatentVariableModel(ABC):
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def reconstruct(self, data: np.ndarray) -> np.float32:
        return self.inverse_transform(self.transform(data))


class PCAPipeline(LatentVariableModel):
    def __init__(self, n_components):
        # TODO: Include flatten + inverse in the pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('drift_model', PCA(n_components=n_components)),
        ])

    def transform(self, data):
        return self.model.transform(data)

    def fit(self, data):
        self.model.fit(data)
        return self

    def explained_variance_ratio(self):
        return self.model[-1].explained_variance_ratio_

    def inverse_transform(self, data):
        return self.model.inverse_transform(data)

    def save(self, filename):
        joblib.dump(self.model, filename)

    @staticmethod
    def load(filename):
        model = joblib.load(filename)
        pca_pipeline = PCAPipeline(n_components=None)
        pca_pipeline.model = model
        return pca_pipeline


class DriftModel:
    def __init__(self, drift_model: LatentVariableModel):
        self.model = drift_model

    def predict(self, images: List[np.ndarray]) -> np.ndarray:
        img_arr = np.array([
            resize(image, RESIZE_DIM)
            for image in images
        ]).reshape(-1, RESIZE_DIM[0] * RESIZE_DIM[1])
        reconstructions = self.model.reconstruct(img_arr)
        drift_score = mean_squared_error(img_arr, reconstructions)
        return drift_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.train:
        # TODO: Use cmd line args to split up training and testing
        # Fit true distribution
        flattened_dim = RESIZE_DIM[0] * RESIZE_DIM[1]
        true_distribution_images = np.array([
            resize(np.array(Image.open(img)), RESIZE_DIM)
            for img in LocalFileSystem().glob("train/**png")
        ]).reshape(-1, flattened_dim)
        dataset_size = true_distribution_images.shape[0]

        # Find optimal number of components
        evr_thres = 0.95
        model = PCAPipeline(min(flattened_dim, dataset_size))
        model.fit(true_distribution_images)
        ev_cumsum = np.cumsum(model.explained_variance_ratio())
        ev_index = np.abs(ev_cumsum - evr_thres).argmin()

        # Fit Drift model
        # Note: If the images are too large, then we could use IncrementalPCA or
        #   resize them to a smaller shape.
        model = PCAPipeline(n_components=ev_index + 1).fit(true_distribution_images)
        model.save(DRIFT_PICKLE_FILE)
    else:
        # Predict drift score
        test_distribution_images = [
            np.array(Image.open(img))
            for img in LocalFileSystem().glob("test/**png")
        ]
        batches = toolz.partition_all(10, test_distribution_images)
        model = PCAPipeline.load(DRIFT_PICKLE_FILE)
        drift_model = DriftModel(model)
        for batch in batches:
            score = drift_model.predict(batch)
            print(score)
