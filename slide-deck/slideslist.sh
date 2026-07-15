#!/bin/bash

SLIDES_DIR="slides"
OUTPUT_FILE="$SLIDES_DIR/slideslist.json"

# Find all HTML files, sort naturally, strip the directory prefix
files=$(find "$SLIDES_DIR" -maxdepth 1 -type f -name "*.html" \
    | sed "s|^$SLIDES_DIR/||" \
    | sort -V)

{
    echo "["
    first=true
    while IFS= read -r file; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        printf '  "%s"' "$file"
    done <<< "$files"
    echo
    echo "]"
} > "$OUTPUT_FILE"

echo "Generated $OUTPUT_FILE"