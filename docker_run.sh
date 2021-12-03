#!/bin/bash
set -e
set -u
set -o pipefail

IMAGE_NAME="capauto"
OPT_RM="--rm"
GPUS="--gpus all"
OPT_TF="--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"
OPT_DOCKER="--user $(id -u):$(id -g)"
DATA=""
APP_DIR="-v$(cd "$(dirname "$0")" && pwd -P):/app/codes"

usage()
{
  echo "Usage: $(basename /$0) [-k | --keep] [-h | --help] [-n | --name IMAGE_NAME] [-d | --data DATA]"
  echo
  echo "ARGUMENTS:"
  echo -e "  -k, --keep            Keep the container while exit. Use 'docker container start -i' to run it again."
  echo -e "  -n, --name IMAGE_NAME Provide the name of the image. Default is 'capauto'."
  echo -e "  -d, --data PATH       Provide additional absolute path to map into /app/data in the container."
  echo -e "  -h, --help            To show this information."
  echo
  echo "Author: Avan Suinesiaputra - KCL 2021"

  exit 2
}

PARSED_ARGUMENTS=$(getopt -o khn:d: --long keep,help,name:,data: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -k | --keep) OPT_RM="";   shift   ;;
    -n | --name) IMAGE_NAME="$2"; shift 2 ;;
    -h | --help) usage ;;
    -d | --data) DATA="-v$2:/app/data";  shift 2 ;;
    --) shift; break ;;
    *) echo "Unexpected option: $1."
       usage ;;
  esac
done

ACTION="docker run -it ${GPUS} ${OPT_TF} ${OPT_DOCKER} ${OPT_RM} --name capauto ${DATA} ${APP_DIR} ${IMAGE_NAME}"

if test "$#" -ne 0; then
  echo "Unexpected arguments found."
  usage
fi

#set -x
echo $ACTION
eval $ACTION
