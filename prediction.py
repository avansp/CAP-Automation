import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from pathlib import Path
import numpy as np
from dicom_lite import DicomLite
import typer


class Prediction:
    """
    Baseline for ViewPrediction & ESPrediction
    """
    def __init__(self, model_filename: Path):
        # load the model
        self.model = tf.keras.models.load_model(model_filename)

    @staticmethod
    def preprocess(img):
        # format image into tensor, standardized to 0-255
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(tf.expand_dims(img, 2), (224, 224))
        img = tf.image.grayscale_to_rgb(img)

        # standardize
        img = img / np.max(img)
        img = img * 255.

        return img

    def predict(self, img):
        return self.model.predict(tf.expand_dims(self.preprocess(img), axis=0))


class ViewPrediction(Prediction):
    """
    Predicting slice view
    """
    CLASSES = sorted(['SA', '4CH', '2CH RT', 'RVOT', 'OTHER', '2CH LT', 'LVOT'], key=str)

    def __init__(self, model_filename: Path):
        super(ViewPrediction, self).__init__(model_filename)

    def predict_image(self, img: np.ndarray):
        """
        Make prediction on a single image

        Copied from batch_predict from CAP View Prediction.ipynb
        """

        pred = super().predict(img)
        pred = tf.argmax(pred, axis=-1)
        pred_view = self.CLASSES[int(pred)]

        return pred_view

    def predict_dataset(self, dataset: tf.data.Dataset):
        """
        Predicting based on a dataset

        Copied from batch_predict from CAP View Prediction.ipynb
        """
        pred = self.model.predict(dataset)
        pred = tf.argmax(pred, axis=-1)

        return [self.CLASSES[int(x)] for x in pred]

    def predict(self, dcm_filename: Path):
        """
        Predicting view of a DICOM image file(s).
        """
        dcm = DicomLite(dcm_filename)

        results = {
            'PatientID': [],
            'SeriesInstanceUID': [],
            'SeriesDescription': [],
            'NumOfImages': [],
            'PredictedView': [],
            'Confidence': []
        }

        # prediction is for each series
        series = dcm.get_series()
        with typer.progressbar(series, label=f"Predicting {len(series)} series") as progress:
            for ser in progress:
                # get the series rows
                subset = dcm.headers[dcm.headers.SeriesInstanceUID == ser]

                # create list of preprocessed numpy arrays
                images = [Prediction.preprocess(x) for x in subset['Image'].values]

                # create dataset
                ds = tf.data.Dataset.from_tensor_slices(images)
                ds = (ds.batch(16).prefetch(tf.data.experimental.AUTOTUNE))

                # prediction
                views = self.predict_dataset(ds)

                # find unique predictions and confidence for that series
                u, count = np.unique(views, return_counts=True)
                count_sort_ind = np.argsort(-count)
                view = u[count_sort_ind][0]
                conf = np.round(np.max(count) / np.sum(count), 2)

                # append to results
                results['PatientID'].append(subset['PatientID'].iloc[0])
                results['SeriesInstanceUID'].append(ser)
                results['SeriesDescription'].append(subset['SeriesDescription'].iloc[0])
                results['NumOfImages'].append(len(images))
                results['PredictedView'].append(view)
                results['Confidence'].append(conf)

        return pd.DataFrame.from_dict(results)
