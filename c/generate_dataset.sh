#!/usr/bin/env bash

entries=(1 10 100 1000 10000)

for i in "${entries[@]}"; do
	for _ in $(seq "$i"); do
		cat data/seeds_dataset.txt >>"data/$i.txt"
	done
done
