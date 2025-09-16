#!/bin/bash

DATASET=$1

if [ -z "$DATASET" ]; then
  echo "Usage: ./run_sansa.sh [dataset]"
  echo "Available datasets: ml-20m, netflix-prize, msd"
  exit 1
fi

echo "Running SANSA with and without log-sigmoid weighting on dataset: $DATASET"

for WEIGHTED in false true
do
  for DENSITY in 0.005 0.01 0.05
  do
    echo "-------------------------------------"
    echo "Running configuration | weighted=$WEIGHTED | density=$DENSITY"

    # Initialize defaults
    LAMBDA=0
    WBETA=0.0
    SCANS=1
    FINETUNE=5

    # Set hyperparameters per dataset and weighting
    if [ "$DATASET" == "ml-20m" ]; then
      if [ "$WEIGHTED" == true ]; then
        LAMBDA=700; WBETA=0.7; SCANS=5; FINETUNE=10
      else
        LAMBDA=500; SCANS=5; FINETUNE=10
      fi

    elif [ "$DATASET" == "netflix-prize" ]; then
      if [ "$WEIGHTED" == true ]; then
        LAMBDA=1600; WBETA=0.8; SCANS=4; FINETUNE=10
      else
        LAMBDA=1000; SCANS=4; FINETUNE=10
      fi

    elif [ "$DATASET" == "msd" ]; then
      if [ "$WEIGHTED" == true ]; then
        LAMBDA=200; WBETA=0.1; SCANS=4; FINETUNE=10
      else
        LAMBDA=200; SCANS=4; FINETUNE=10
      fi

    else
      echo "Unknown dataset: $DATASET"
      exit 1
    fi

    CMD="python3 sansa.py --dataset $DATASET --lambda $LAMBDA --density $DENSITY --scans $SCANS --finetune $FINETUNE"
    if [ "$WEIGHTED" == true ]; then
      CMD="$CMD --wflg --wbeta $WBETA"
    fi

    echo "$CMD"
    eval $CMD
  done
done
