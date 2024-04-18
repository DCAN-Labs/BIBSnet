#!/bin/bash

set -eu

DIR=$1

for TOOL in `ls $DIR/bin`; do
  if [[ $(head -n1 $DIR/bin/$TOOL) == *"python" ]]; then
    sed -i '1 c#! /usr/bin/env python' $DIR/bin/$TOOL && echo "Fixed: $DIR/bin/$TOOL";
  fi
done
