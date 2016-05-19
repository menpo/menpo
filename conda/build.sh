#!/bin/sh

cp -r $RECIPE_DIR/../ .

"$PYTHON" setup.py install --single-version-externally-managed --record=/tmp/record.txt
