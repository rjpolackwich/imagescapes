#!/usr/bin/env bash

set -e

pip3 install --upgrade pip
pip3 install --ignore-installed -r build/chipper_requirements.txt

# sync data
aws s3 sync s3://imagesim-storage/ ~/data/
