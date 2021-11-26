import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd


class DicomLite:
    """
    Extract necessary DICOM header
    """
    def __init__(self, dcm_filename: Path):
        assert dcm_filename.is_file(), f"File {dcm_filename} does not exist."

        ds = pydicom.read_file(dcm_filename, force=True)

        # get patient, study, and series information
        self.patientID = DicomLite.clean_text(ds.get("PatientID", "NA"))
        self.studyDescription = DicomLite.clean_text(ds.get("StudyDescription", "NA"))
        self.seriesDescription = DicomLite.clean_text(ds.get("SeriesDescription", "NA"))

        # generate new, standardized file name
        self.modality = ds.get("Modality", "NA")
        self.studyInstanceUID = ds.get("StudyInstanceUID", "NA")
        self.seriesInstanceUID = ds.get("SeriesInstanceUID", "NA")
        self.seriesNumber = ds.get('SeriesNumber', 'NA')
        self.instanceNumber = str(ds.get("InstanceNumber", "0"))

        # load image data - preprocess it
        self.image = ds.pixel_array

    @staticmethod
    def clean_text(string):
        # clean and standardize text descriptions, which makes searching files easier
        forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
        for symbol in forbidden_symbols:
            string = string.replace(symbol, "_")  # replace everything with an underscore

        return string.lower()


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

    def batch_predict(self, path: Path):
        """
        Batch prediction from a folder
        """
        # dicom files can be with *.dcm or not; so let's grab all of them
        for f in path.rglob('*'):
            if not f.is_file():
                continue


class ViewPrediction(Prediction):
    """
    Predicting slice view
    """
    CLASSES = sorted(['SA', '4CH', '2CH RT', 'RVOT', 'OTHER', '2CH LT', 'LVOT'], key=str)

    def __init__(self, model_filename: Path):
        super(ViewPrediction, self).__init__(model_filename)

    def predict_file(self, dcm_filename: Path):
        """
        Predicting view of a DICOM image file.
        """
        ds = DicomLite(dcm_filename)
        pred = tf.argmax(self.predict(ds.image), axis=-1)

        return self.CLASSES[int(pred)]

