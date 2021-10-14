cd train_code/resnext50
for i in {0..4};do
  python train.py --fold $i --nfolds 5 --batch_size 4 --epochs 15 --lr 1e-4 --expansion 32 --workers 1 --path ../../data --weight_decay 1e-5 --transfer 1
done
cd ../resnext101
for i in {0..4};do
  python train.py --fold $i --nfolds 5 --batch_size 4 --epochs 15 --lr 1e-4 --expansion 32 --workers 1 --path ../../data --weight_decay 1e-5 --transfer 1
done
cd ../../
