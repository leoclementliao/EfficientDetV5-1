This is a training scheme for WHEAT DETECTION.

add:
1) mosaic + mixup with original cutout and random crop and flip brightness hue etc.
2) batch accumulation , nominal batchsize = 64
3) weight decay for all none bn weights
4) multi-card traning is not fully functional, suggest using single card
5) progress bar when training.
6) saving model using half precision, lower model size
7) Exponential moving average weight support

to add:
1) anchor ration to change to yolo style
2) add MAP calculation for validation set
3) loss change to yolo style

