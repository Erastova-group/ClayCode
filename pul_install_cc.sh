#!/bin/bash

cd ClayCode
git pull
bash install.sh
ClayCode builder -f /storage/claycode/package/ClayCode/builder/tests/data/input.yaml
vmd /storage/clay_models/NAu-1-fe/NAu-1-fe_7_5_solv_ions.gro
