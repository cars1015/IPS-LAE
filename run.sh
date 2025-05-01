#!/bin/bash

DATASET=$1

if [ -z "$DATASET" ]; then
  echo "Usage: ./run.sh [dataset]"
  echo "Available datasets: ml-20m, netflix-prize, msd"
  exit 1
fi

echo "Running all models (with and without log-sigmoid weighting) on dataset: $DATASET"

for MODEL in ease edlae rdlae
do
  echo "-------------------------------------"
  for WEIGHTED in false true
  do
    if [ "$WEIGHTED" == true ]; then
          WFLAG="--wflg --wtype logsigmoid"
    else
      WFLAG=""
    fi

    # Initialize hyperparameters
    LAMBDA=0
    BETA=0
    DROP_P=0
    ALPHA=0

    #Set values for each (dataset, model, weighting) combination

    if [ "$DATASET" == "ml-20m" ]; then
      if [ "$MODEL" == "ease" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=700; BETA=0.7
        else
          LAMBDA=500
        fi
      elif [ "$MODEL" == "edlae" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=500; BETA=0.7; DROP_P=0.3
        else
          LAMBDA=400; DROP_P=0.3
        fi
      elif [ "$MODEL" == "rdlae" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=00; BETA=0.1; DROP_P=0.4; ALPHA=0.2
        else
          LAMBDA=1000; DROP_P=0.2; ALPHA=0.6
        fi
      fi

    elif [ "$DATASET" == "netflix-prize" ]; then
      if [ "$MODEL" == "ease" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=1600; BETA=0.8
        else
          LAMBDA=1000
        fi
      elif [ "$MODEL" == "edlae" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=1300; BETA=0.75; DROP_P=0.3
        else
          LAMBDA=600; DROP_P=0.3
        fi
      elif [ "$MODEL" == "rdlae" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=1100; BETA=0.7; DROP_P=0.3; ALPHA=0.1
        else
          LAMBDA=700; DROP_P=0.3; ALPHA=0.4
        fi
      fi

    elif [ "$DATASET" == "msd" ]; then
      if [ "$MODEL" == "ease" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=500; BETA=0.2
        else
          LAMBDA=200
        fi
      elif [ "$MODEL" == "edlae" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=100; BETA=0.1; DROP_P=0.1
        else
          LAMBDA=70; DROP_P=0.3
        fi
      elif [ "$MODEL" == "rdlae" ]; then
        if [ "$WEIGHTED" == true ]; then
          LAMBDA=100; BETA=0.1; DROP_P=0.1; ALPHA=0.3
        else
          LAMBDA=80; DROP_P=0.2; ALPHA=0.4
        fi
      fi

    else
      echo "Unknown dataset: $DATASET"
      exit 1
    fi

    CMD="python3 main.py --dataset $DATASET --model $MODEL --lambda $LAMBDA $WFLAG"
    if [ "$WEIGHTED" == true ]; then
      CMD="$CMD --wbeta $BETA"
    fi
    if [ "$MODEL" != "ease" ]; then
      CMD="$CMD --drop_p $DROP_P"
    fi
    if [ "$MODEL" == "rdlae" ]; then
      CMD="$CMD --alpha $ALPHA"
    fi
    eval $CMD
  done

done

