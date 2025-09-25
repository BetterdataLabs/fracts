#!/bin/bash
for i in {4..13}
do
    python main.py -t \
        -o /root/synthetic-ts-long/outputs/ablations/baseball_experiment_$i \
        -d /root/synthetic-ts-long/baseball/encoded_split/train \
        -s /root/synthetic-ts-long/baseball/encoded_split/train.csv \
        -i players_id \
        -c /root/synthetic-ts-long/FracTS/config/baseball_ablations/experiment$i.yaml 
    python main.py -g \
        -o /root/synthetic-ts-long/outputs/ablations/baseball_experiment_$i \
        -c /root/synthetic-ts-long/FracTS/config/baseball_ablations/experiment$i.yaml \
        -n 826 -a
done