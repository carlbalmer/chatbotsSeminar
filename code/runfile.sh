#!/bin/bash

export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64

cd ~/Documents/Uni/chatbotsSeminar/code

for (( i=3; i <= 10; i=i+1 )); do
  	P=$(python -c "print $i/10.0")
  	echo "$P"

  	mkdir ./data/train/
  	cp ./data/test.csv ./data/train/test.csv
  	cp ./data/valid.csv ./data/train/valid.csv

  	python3 ./scripts/noise_data.py --noise_probability "$P"
  	python3 ./scripts/prepare_data.py --input_dir ./data/train --output_dir ./data/train

  	mkdir "./runs/$P"
  	python3 ./udc_train.py --input_dir ./data/train --model_dir "./runs/$P" --num_epochs "40000" &> "./runs/$P/log.txt"

  	rm -rf ./data/train

done
