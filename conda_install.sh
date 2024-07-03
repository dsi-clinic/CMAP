#!/bin/bash

# Path to the text file containing package names
packages_file="requirements.txt"

# Check if the file exists
if [ ! -f "$packages_file" ]; then
    echo "Error: File $packages_file not found."
    exit 1
fi

# Read the file line by line and install each package using conda
while IFS= read -r package; do
    conda install --yes "$package"
done < "$packages_file"

