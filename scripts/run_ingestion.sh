#!/bin/bash

# Set the start date
start_date="20240101"

# Get end date: today
end_date=$(date +%Y%m%d)

# Run the Python script
python -m ingestion.get_data -s $start_date -e $end_date --save_raw_data

# Zip the raw json files
find test-data -type f -name "*.json" | zip -m raw_data.zip -@