#!/bin/bash

# This script ensures that the data directory exists and has the right permissions
# It should be run before starting the container

# Create directories if they don't exist
mkdir -p data
mkdir -p models
mkdir -p logs

# Set permissions
chmod -R 777 data
chmod -R 777 models
chmod -R 777 logs

echo "Data directories created and permissions set"
