#!/bin/bash
python evaluate.py --data_path 'data/test/PWMAXSAT/3CNF/*.wcnf' --model_dir models/PWMAXSAT --num_boost 1 --timeout 1200 --verbose