#!/bin/bash
stime=$(date +%s)
cd train_code/resnext50
for i in {0..4};do
  python train.py --fold $i --nfolds 5 --batch_size 16 --epochs 50 --lr 1e-5 --expansion 32 --workers 1 --path ../../data --transfer 0
done
cd ../resnext101
for i in {0..4};do
  python train.py --fold $i --nfolds 5 --batch_size 16 --epochs 50 --lr 1e-5 --expansion 32 --workers 1 --path ../../data --transfer 0
done
cd ../../
etime=$(date +%s)
echo "Time Elapsed: $(($etime-$stime)) seconds."