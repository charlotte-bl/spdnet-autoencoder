#!/bin/bash

encoding_dims=(2 4 6 8 10 12 14 16)
encoding_channels=(1 2 3 4 5 6 7 8)
xp=4
cd src || exit 1

for i in $(seq 1 $xp); do
    for dim in "${encoding_dims[@]}"; do
		for channel in "${encoding_channels[@]}"; do
			echo "XP $i :"
			echo " | Parameters : encoding_dim = $dim , encoding_channel = $channel , "
			python3 pipeline.py --encoding_dim "$dim" --epochs 200 --data bci --encoding_channel "$channel" --loss riemann --layers_type by_halves 
		done
    done
done

cd ..