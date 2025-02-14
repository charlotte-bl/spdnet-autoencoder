#!/bin/bash

encoding_dims=(2 4 6 8)
encoding_channels=(1 2 4 8)
xp=2
datasets=(2)
cd src || exit 1

for i in $(seq 1 $xp); do
    for dim in "${encoding_dims[@]}"; do
		for channel in "${encoding_channels[@]}"; do
			for dataset in "${datasets[@]}"; do
				echo "XP $i :"
				echo " | Parameters : encoding_dim = $dim"
				echo " || Dataset : index = $dataset"
				python3 pipeline.py --encoding_dim "$dim" -t geodesics --encoding_channel "$channel" --epochs 150 --loss riemann 
			done
		done
    done
done

cd ..
