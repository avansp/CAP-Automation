import typer
from pathlib import Path
from prediction import ViewPrediction, Prediction
import json
from enum import Enum

app = typer.Typer()
app_config = dict()


class ViewNetwork(str, Enum):
    resnet50 = "resnet50",
    vgg19 = "vgg19",
    xception = "xception"


@app.command()
def view(path: Path = typer.Argument(None, help="A path of a DICOM image or a folder"),
         model_name: ViewNetwork = typer.Option("resnet50", "--model", help="Network model name", case_sensitive=False)):
    """
    Automatically predict slice view of a DICOM image or all DICOM images in a folder.
    """
    if model_name.value not in app_config["view_selection"]["models"]:
        typer.echo(f"Network model {model_name.value} is not listed in the list of slice view selection models "
                   f"in the configuration file.")
        typer.Abort()

    typer.echo(f"Slice view prediction using {model_name.value}")
    model_filename = Path(app_config["view_selection"]["model_folder"]) / app_config["view_selection"]["models"][model_name.value]

    if not model_filename.is_file():
        typer.echo(f"Cannot read {model_filename}.")
        typer.Abort()

    pred = ViewPrediction(model_filename)

    if path.is_file():
        typer.echo(f"Predicting a single image file: {path}")
        view_name = pred.predict_file(path)
        typer.echo("Predicted view is: " + typer.style(f"{view_name}", fg=typer.colors.GREEN, bold=True))
    elif path.is_dir():
        typer.echo(f"Predicting a folder: {path}")
        # pred.batch_predict(path)
    else:
        typer.echo(f"Input path {path} is neither a file or a folder.")
        typer.Abort()


@app.command()
def model(model_filename: Path):
    """
    Inspect a model
    """
    typer.echo(f"Reading model: {model_filename}")
    pred = Prediction(model_filename)

    typer.echo(pred.model.summary())


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
