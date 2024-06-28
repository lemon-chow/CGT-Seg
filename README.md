# Seg_lung

```
train.py -c 4 -e 8 -b 16 -l 1e-4 -lw 1.0 2.6 -cw 1 1.7 2.4 2.8
predict.py -m checkpoints/checkpoint_epoch8.pth -c 4 -i /root/LLUNET/data/image_1/25/100.png -o res.png

python predict.py -m checkpoints/checkpoint_epoch15.pth -c 4 -i /root/LLUNET/data/image_1/25/100.png -o res.png
UResnet:
1: python train.py -c 4 -e 8 -b 16 -l 1e-4 -lw 1.0 2.6 -cw 1 1.7 2.4 2.8
2: python train.py -c 4 -e 8 -b 16 -l 1e-4 -lw 1.0 2.0 -cw 1 1.7 2.4 4
3: python train.py -c 4 -e 8 -b 16 -l 1e-4 -lw 1.0 8.8 -cw 1 1.7 2.4 4.5
4: python train.py -c 4 -e 8 -b 16 -l 1e-4 -lw 1.0 5 -cw 1 1.7 2.4 4.5
5: python train.py -c 4 -e 8 -b 16 -l 1e-4 -lw 1.0 4.5 -cw 1 1.7 2.4 5
6: python train.py -c 4 -e 16 -b 16 -l 1e-4 -lw 1.0 4.5 -cw 1 1.7 2.4 5
7: python train.py -c 4 -e 16 -b 16 -l 1e-4 -lw 1.0 4.5 -cw 1 2.0 2.4 5
8: python train.py -c 4 -e 16 -b 16 -l 1e-4 -lw 1.0 4.7 -cw 1 2.0 3.0 5
9: python train.py -c 4 -e 16 -b 16 -l 1e-4 -lw 1.0 4 -cw 1 2.3 3.0 5
```
