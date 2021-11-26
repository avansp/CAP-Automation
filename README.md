Towards fully automated cardiac statistical modeling: a deep-learning based MRI view and frame selection tool
----

*This is forked from https://github.com/btcrabb/CAP-Automation*


## Usage

1. Build a docker image from scratch:
```bash
$ docker build -t capauto .
```

2. Run the image above as a container:
```bash
$ ./docker_run.sh
```

If you have stored your models and images in a folder called `DATA`, then you can run the image as
```bash
$ ./docker_run.sh --data DATA
```
It will be mapped into `/app/data/` in the container.

See `./docker_run.sh --help` for more options

## More information

See the original repository by Brendan Crabb: [CAP-Automation](https://github.com/btcrabb/CAP-Automation)

