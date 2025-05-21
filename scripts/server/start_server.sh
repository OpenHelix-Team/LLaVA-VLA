#!/bin/bash

which python
# export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

python llava/serve/flask_helper.py & python llava/serve/flask_server.py
