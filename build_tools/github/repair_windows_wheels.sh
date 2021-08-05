#!/bin/bash

# TODO: This file is adapted from scikit-learn

set -e
set -x

WHEEL=$1
DEST_DIR=$2
BITNESS=$3

# By default, the Windows wheels are not repaired.
# In this case, we need to vendor VCRUNTIME140.dll
wheel unpack "$WHEEL"
WHEEL_DIRNAME=$(ls -d gensvm-*)

ls -R -a "${WHEEL_DIRNAME}"

python build_tools/github/vendor.py "$WHEEL_DIRNAME" "$BITNESS"

sleep 1

ls -R -a "${WHEEL_DIRNAME}"

sleep 1

cat "${WHEEL_DIRNAME}/gensvm/_distributor_init.py"

sleep 1

ldd "${WHEEL_DIRNAME}/gensvm/cython_wrapper/wrapper.cp37-win32.pyd"

sleep 1

wheel pack "$WHEEL_DIRNAME" -d "$DEST_DIR"
rm -rf "$WHEEL_DIRNAME"
