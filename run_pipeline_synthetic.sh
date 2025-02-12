#!/bin/bash

encoding_dims=(2 4 6 8 10 12 14 16)
xp=1
datasets=(1 2 3 4 5)
cd src || exit 1

for i in $(seq 1 $xp); do
    for dim in "${encoding_dims[@]}"; do
		for dataset in "${datasets[@]}"; do
			echo "XP $i :"
			echo " | Parameters : encoding_dim = $dim"
			echo " || Dataset : index = $dataset"
        	python3 pipeline.py --encoding_dim "$dim" --epochs 50 --index "$dataset" --encoding_channel 1 --loss euclid
		done
    done
done

cd ..
