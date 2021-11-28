Towards fully automated cardiac statistical modeling: a deep-learning based MRI view and frame selection tool
----

*This is forked from https://github.com/btcrabb/CAP-Automation*

> #### DISCLAIMER
> 
> I am only converting [the original repo](https://github.com/btcrabb/CAP-Automation) codes into usable implementation for predicting slice view. I did not develop the codes. Results are presented as is. If you have questions about the method or the results, please ask the original author.
> 
> -- *avansp* --
>

## Configuration

Assume that network models, input images and results are stored in `/path/to/your/data`.

1. Copy network models for view prediction from https://github.com/btcrabb/CAP-Automation to `/path/to/your/data/models/view_selection`

2. Copy network models for ES phase selection from https://github.com/btcrabb/CAP-Automation to `/path/to/your/data/models/es_selection`

3. Copy input CMR studies to `/path/to/your/data/images/`

Note that if you use different folders for network models, you must alter the [config.json](./config.json) file. In the config.json file `/app` is the root in the container that maps to `/path/to/your/data` (see running a container section below).

## Running a container

*Note that I'm using a docker from TensorFlow, which was built using older CUDA driver (11.0). This might slow the prediction process. You can build a docker using newer CUDA driver to optimise your GPU usage.*

1. Build a docker image:
```bash
$ docker build -t capauto .
```

2. Run the image above as a container:
```bash
$ ./docker_run.sh --data /path/to/your/data
```

The above script will run a container and map two folders:

* `/app/codes` to the codes in your host filesystem
* `/app/data` to `/path/to/your/data` folder in your host filesystem

You can use `/path/to/your/data` folder to store input images, models and results.

## View prediction

Assume you store the input CMR study into `/path/to/your/data/images/YOUR_CMR_STUDY`. 

From inside the container you can run a prediction as

```
tf-docker /app > cd codes
tf-docker /app/codes > python predict.py view --output /app/results/prediction_results.csv \
/app/data/images/YOUR_CMR_STUDY
```

For more options, call `python predict.py --help`

## Requirements

If you don't want to use docker, then here are required packages to run the codes:

* Python >= 3.6
* [Tensorflow](https://www.tensorflow.org/) v. 2.4.1
* [Typer](https://typer.tiangolo.com)
* [Pydicom](https://github.com/pydicom/pydicom)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)

## More information

See the original repository by Brendan Crabb: [CAP-Automation](https://github.com/btcrabb/CAP-Automation)

