#!/bin/bash

for file in examples/*.tensorforth; do
	if [ -f "$file" ]; then 
		echo "Running" "$file"
		"./tensorforth" "$file"
	fi
done