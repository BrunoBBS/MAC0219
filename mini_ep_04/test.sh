#!/bin/bash


number_th=(5 8 10)
number_ac=(10000 100000 3000000)
scheds=("SCHED_OTHER" "SCHED_FIFO" "SCHED_RR")

for sched in ${scheds[@]}; do
    sed -i "s/#define OS_SCHEDULER .*/#define OS_SCHEDULER $sched/" main.c
    for sync in "\/\/" ""; do
        sed -rie "s/(\/\/|)__sync_synchronize\(\);/${sync}__sync_synchronize\(\);/g" gate.c
        sed -rie "s/(\/\/|)__sync_synchronize\(\);/${sync}__sync_synchronize\(\);/g" bakery.c
        make
        for th in ${number_th[@]}; do
            for ac in ${number_ac[@]}; do
                echo "s/(\/\/|)__sync_synchronize\(\);/${sync}__sync_synchronize\(\);/g"
                echo SCHEDULER: $sched
                echo THREADS: $th
                echo ACCESS: $ac
                echo SYNC: $sync
                ./main $th $ac
            done
        done
    done
done
