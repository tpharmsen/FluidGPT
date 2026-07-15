#!/bin/bash

# Usage:
# ./latex_to_svg.sh input.tex output.svg

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.tex output.svg"
    exit 1
fi

TEX_FILE="$1"
SVG_FILE="$2"

# Get filename without extension
BASE_NAME="${TEX_FILE%.tex}"

# Compile LaTeX to DVI
latex -latex "$TEX_FILE"

if [ $? -ne 0 ]; then
    echo "LaTeX compilation failed."
    exit 1
fi

# Convert DVI to SVG
#dvisvgm --font-format=woff "${BASE_NAME}.dvi" -o "$SVG_FILE"
dvisvgm --no-fonts "${BASE_NAME}.dvi" -o "$SVG_FILE"

if [ $? -ne 0 ]; then
    echo "SVG conversion failed."
    exit 1
fi

# Replace colors
sed -i \
    -e 's/#000000/#ffffff/gI' \
    -e 's/#1c1c1c/#ffffff/gI' \
    -e 's/#000/#ffffff/gI' \
    "$SVG_FILE"

echo "Created: $SVG_FILE"