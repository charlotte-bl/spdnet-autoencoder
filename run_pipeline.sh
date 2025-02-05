#!/bin/bash

latent_dims=(2 4 6 8 10 12 14 16)

cd src || exit 1

for dim in "${latent_dims[@]}"; do
	echo "XP - Parameters : latent_dim = $dim"
	python3 pipeline.py --latent_dim "$dim" --epochs 50
done

cd -
