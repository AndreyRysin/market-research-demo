#!/bin/bash

# paths
DATA_EXEC=../src/data_exec
ML_EXEC=../src/ml_exec

# functions
printstep(){
    echo
    echo "============================="
    echo "=========  STEP $1  ========="
    echo "============================="
}

# options
while getopts "abc" OPTION; do
    case $OPTION in
        a)
        OPT_A=true ;;
        b)
        OPT_B=true ;;
        c)
        OPT_C=true ;;
        ?)
        echo "enabled options:"
        echo "[-a], [-b], [-c]"
        exit ;;
    esac
done

# preprocess raw FX data [-a]
if [ "$OPT_A" = true ]; then
    printstep 00 && python3 $DATA_EXEC/create_data_folders.py --data_cfg a || exit
    printstep 01 && python3 $DATA_EXEC/preprocess.py --data_cfg a || exit
fi

# extract FX features [-b]
if [ "$OPT_B" = true ]; then
    printstep 02 && python3 $DATA_EXEC/extract_primary.py --mode features --data_cfg b --prim_pipe b || exit
    printstep 03 && python3 $DATA_EXEC/extract_primary.py --mode targets --data_cfg b --prim_pipe b || exit
    printstep 04 && python3 $DATA_EXEC/select_primary_features.py --data_cfg b || exit
    printstep 05 && python3 $DATA_EXEC/extract.py --mode features --data_cfg b || exit
    printstep 06 && python3 $DATA_EXEC/check_identical.py --mode features --data_cfg b || exit
fi

# extract FX targets [-c]
if [ "$OPT_C" = true ]; then
    printstep 07 && python3 $DATA_EXEC/extract_primary.py --mode targets --data_cfg c --prim_pipe c || exit
    printstep 08 && python3 $DATA_EXEC/extract.py --mode targets --data_cfg c || exit
    printstep 09 && python3 $DATA_EXEC/check_identical.py --mode targets --data_cfg c || exit
fi
