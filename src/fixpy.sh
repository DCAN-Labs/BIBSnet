#!/bin/bash

set -eu

FSLDIR=$1

for TOOL in `ls $FSLDIR/bin`; do
  if [[ $(head -n1 $TOOL) == *"python" ]]; then
    sed -i '1 c#! /usr/bin/env python' $TOOL && echo "Fixed: $TOOL";
  fi
done