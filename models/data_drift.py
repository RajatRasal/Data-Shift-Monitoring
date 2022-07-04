import numpy as np
from typing import List

import toolz
from PIL import Image
from fsspec.implementations.local import LocalFileSystem
from joblib import dump, load
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


DRIFT_PICKLE_FILE = "data_drift_model.pickle"
RESIZE_DIM = (220, 220)


class PCADriftModel:
    def __init__(self):
        self.model = load(DRIFT_PICKLE_FILE)

    def predict(self, images) -> np.float32:
        img_arr = np.array([
            resize(image, RESIZE_DIM)
            for image in images
        ]).reshape(-1, RESIZE_DIM[0] * RESIZE_DIM[1])
        projection = self.model.transform(img_arr)
        reconstructions = self.model.inverse_transform(projection)
        drift_score = mean_squared_error(img_arr, reconstructions)
        return drift_score


if __name__ == "__main__":
    # Fit true distribution
    true_distribution_images = np.array([
        resize(np.array(Image.open(img)), RESIZE_DIM)
        for img in LocalFileSystem().glob("train/**png")
    ]).reshape(-1, RESIZE_DIM[0] * RESIZE_DIM[1])
    model = PCA(n_components=50).fit(true_distribution_images)
    dump(model, DRIFT_PICKLE_FILE)

    # Predict drift score
    test_distribution_images = [
        np.array(Image.open(img))
        for img in LocalFileSystem().glob("test/**png")
    ]
    batches = toolz.partition_all(10, test_distribution_images)
    drift_model = PCADriftModel()
    for batch in batches:
        score = drift_model.predict(batch)
        print(score)
