#!/usr/bin/env bash

set -e

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda

pip3 install --upgrade pip
pip3 install --ignore-installed -r build/requirements.txt

# TODO: sync dataset splits
