#!/bin/bash

# Project root (override: export QR_ROOT=/path/to/QueryRobustness)
QR_ROOT="${QR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# mkdir -p data/dpr
# wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# pushd data/dpr
# gzip -d psgs_w100.tsv.gz
# popd

# cd data
# wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
# mv ./../elasticsearch-7.17.9.tar.gz .
# tar zxvf elasticsearch-7.17.9.tar.gz
# # rm elasticsearch-7.17.9.tar.gz  # if need space
# cd elasticsearch-7.17.9
cd ${QR_ROOT}/datasets/data/elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
echo "Elasticsearch is running"