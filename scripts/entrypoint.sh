#!/bin/bash

CHECKPOINT_FILE="/checkpoints/last.ckpt"
DATA_DIR="/data/processed"

if [ -f $CHECKPOINT_FILE ]; then
    echo "Checkpoint file found. Restoring from checkpoint..."
    exec ./scripts/run_simulation.sh
else
    echo "Checkpoint file not found. Checking for data..."
    if [ -d $DATA_DIR ]; then
        echo "Data found. Training model..."
        exec ./scripts/run_training.sh
        echo "Running simulation..."
        exec ./scripts/run_simulation.sh
    else
        echo "Data not found. Running data ingestion script..."
        exec ./scripts/run_ingestion.sh
        echo "Training model..."
        exec ./scripts/run_training.sh
        echo "Running simulation..."
        exec ./scripts/run_simulation.sh
fi
