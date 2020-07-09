#!/usr/bin/env bash

set -e

rm -rf lightning_logs || true
rm *.chkpt || true

CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
python3 model/run_model.py --model resnet --data_path ~/data/ --gpus -1 --distributed_backend ddp --max_epochs 100
