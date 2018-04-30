#!/bin/bash
for i in 10000 100000 1000000; do
    for j in 1 10 100 ; do
        for k in $(seq 1 30); do
            echo "================================================="
            echo "running vector size $i, $j threads"
            echo "================================================="
            ./contention.sh $i $j | grep -E "[0-9]\.[0-9]+">> $i$j.txt
        done
    done
done
