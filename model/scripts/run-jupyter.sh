#!/usr/bin/env bash

jupyter notebook \
        --notebook-dir=/workdir \
        --ip 0.0.0.0 \
        --no-browser \
        --allow-root