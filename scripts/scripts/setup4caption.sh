#!/bin/bash
ossutil64 cp -r -f  oss://antsys-vilab/yyb/PlatonicGen/third_party/AutoGPTQ/ ./AutoGPTQ/
cd ./AutoGPTQ && pip install -vvv --no-build-isolation -e . && cd ..
pip install --upgrade pyparsing