import typer
from pathlib import Path
from prediction import ViewPrediction, Prediction
from dicom_lite import DicomLite
import json
from enum import Enum

app = typer.Typer()
app_config = dict()


class ViewNetwork(str, Enum):
    resnet50 = "resnet50",
    vgg19 = "vgg19",
    xception = "xception"


@app.command()
def view(path: Path = typer.Argument(None, help="A path of a DICOM image or a folder."),
         model_name: ViewNetwork = typer.Option("resnet50", "--model", help="Network model name", case_sensitive=False),
         output: Path = typer.Option(None, help="Save results as a CSV file")):
    """
    Automatically predict slice view of a DICOM image or all DICOM images in a folder.
    """
    if model_name.value not in app_config["view_selection"]["models"]:
        typer.echo(f"Network model {model_name.value} is not listed in the list of slice view selection models "
                   f"in the configuration file.")
        typer.Abort()

    # load the network model
    typer.echo(f"Slice view prediction using {model_name.value}")
    model_filename = Path(app_config["view_selection"]["model_folder"]) / app_config["view_selection"]["models"][model_name.value]

    if not model_filename.is_file():
        typer.echo(f"Cannot read {model_filename}.")
        typer.Abort()

    pred = ViewPrediction(model_filename)

    # predict
    results = pred.predict(path)

    # save output
    if output is not None:
        results.to_csv(output)
        typer.secho(f"Results are saved in {output}", fg=typer.colors.MAGENTA, italic=True)

    # print output
    typer.secho(f"Results:", fg=typer.colors.GREEN, bold=True)
    typer.secho(results, fg=typer.colors.GREEN)


@app.command()
def inspect_model(model_filename: Path = typer.Argument(None, help="Neural network model file (in HDF5 format).")):
    """
    Inspect a model
    """
    typer.echo(f"Reading model: {model_filename}")
    pred = Prediction(model_filename)

    typer.echo(pred.model.summary())


@app.command()
def dicom_lite(path: Path = typer.Argument(None, help="A DICOM file or a folder that contains DICOM images.")):
    """
    Test DicomLite class.
    """
    dcm = DicomLite(path)
    typer.echo("Sample header data:")
    typer.echo(dcm.headers.head().transpose())


@app.callback()
def main(config_filename: Path = typer.Option("./config.json", "--config", help="Configuration file (JSON).")):
    """
    CAP-Automation prediction. See the help from each command for more information.
    """
    global app_config

    # read config
    with open(config_filename, 'r') as f:
        app_config = json.load(f)


if __name__ == "__main__":
    app()
