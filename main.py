import typer

app = typer.Typer()


@app.command()
def predict_view():
    typer.echo("Predicting slice view")


if __name__ == "__main__":
    app()
