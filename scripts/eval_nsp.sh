#!/bin/bash
python evaluate.py --data_path 'data/validation/NSP/*.nsp' --model_dir models/NSP --num_boost 10 --network_steps 10000