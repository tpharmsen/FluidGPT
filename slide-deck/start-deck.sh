#!/usr/bin/env bash

set -e

# Generate once initially
./slideslist.sh

# Watch for changes in the background
while inotifywait -e create -e delete -e moved_to -e moved_from slides; do
    ./slideslist.sh
done &

# Start the server
exec npx serve . --listen tcp://127.0.0.1:3000