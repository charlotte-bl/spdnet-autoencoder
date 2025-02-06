#!/bin/bash

latent_dims=(2 4 6 8 10 12 14 16)
xp=5
datasets=(1 2 3 4 5)
cd src || exit 1

for i in $(seq 1 $xp); do
    for dim in "${latent_dims[@]}"; do
		for dataset in "${datasets[@]}"; do
			echo "XP $i :"
			echo " | Parameters : latent_dim = $dim"
			echo " || Dataset : index = $dataset"
        	python3 pipeline.py --latent_dim "$dim" --epochs 50 --index "$dataset"
		done
    done
done

cd ..
