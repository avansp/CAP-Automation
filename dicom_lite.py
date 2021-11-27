import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from pathlib import Path
import typer
import pydicom
import pandas as pd


class DicomLite:
    """
    Extract necessary DICOM header. Store headers as pandas.DataFrame.
    Recursive glob if given a folder as initial argument.
    Image(s) are already preprocessed & converted into tensor.
    """
    IMAGE_SIZE = (224, 224)

    def __init__(self, path: Path):
        files = []
        if path.is_file():
            files = [path]
        elif path.is_dir():
            # dicom files can be with *.dcm or not; so let's grab all of them
            files = [f for f in path.rglob("*") if f.is_file()]
        else:
            typer.echo(f"Input path {path} is neither a file or a folder.")
            typer.Abort()

        # collect DICOM headers
        self.headers = {
            'PatientID': [],
            'Filename': [],
            'StudyDescription': [],
            'SeriesDescription': [],
            'Modality': [],
            'StudyInstanceUID': [],
            'SeriesInstanceUID': [],
            'SeriesNumber': [],
            'InstanceNumber': [],
            'Image': []
        }

        with typer.progressbar(files, label="Collecting DICOM files") as progress:
            for f in progress:
                try:
                    dcm = pydicom.read_file(f, force=True)
                    if "TransferSyntaxUID" not in dcm.file_meta:
                        continue
                except:
                    continue

                for k in self.headers:
                    if k in ['PatientID', 'StudyDescription', 'SeriesDescription']:
                        self.headers[k].append(DicomLite.clean_text(dcm.get(k, "NA")))
                    elif k in ['InstanceNumber']:
                        self.headers[k].append(str(dcm.get(k, "0")))
                    elif k == 'Image':
                        self.headers[k].append(dcm.pixel_array)
                    elif k == 'Filename':
                        self.headers[k].append(f)
                    else:
                        self.headers[k].append(dcm.get(k, "NA"))

        # convert to DataFrame
        self.headers = pd.DataFrame(self.headers)

        typer.echo(f"Found {self.headers.shape[0]} eligible DICOM files.")

    def get_series(self):
        """
        Get a unique set of series UID's from the table
        """
        return list(set(self.headers.SeriesInstanceUID.values))

    @staticmethod
    def clean_text(string):
        """
        Cleaning string from some headers.

        Copied from CAP View Prediction.ipynb
        """
        # clean and standardize text descriptions, which makes searching files easier
        forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
        for symbol in forbidden_symbols:
            string = string.replace(symbol, "_")  # replace everything with an underscore

        return string.lower()

