#!/usr/bin/env bash
SCRIPT_DIR=$(dirname $0)
PYTHONPATH=$SCRIPT_DIR python3 -m cli "$@"