import argparse
import random
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import torch_utils
import yaml
import math
from data.data import *
from compute_loss import *
from tqdm import tqdm
import numpy as np
from utils import *
#import test  # import test.py to get mAP after each epoch

#from timm.utils import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


#from utils.datasets import *
#from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
    #mixed_precision = False
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
os.makedirs(wdir, exist_ok=True)
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters
hyp = {'lr0': 0.0004,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.900,  # SGD momentum
       'weight_decay': 4e-5,#5e-4,  # optimizer weight decay
#       'giou': 0.2,  # giou loss gain
#       'cls': 0.58,  # cls loss gain
#       'cls_pw': 1.0,  # cls BCELoss positive_weight
#       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
#       'obj_pw': 1.0,  # obj BCELoss positive_weight
#       'iou_t': 0.20,  # iou training threshold
#       'anchor_t': 4.0,  # anchor-multiple threshold
#       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
#       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
#       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
#       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
#       'degrees': 0.0,  # image rotation (+/- deg)
#       'translate': 0.0,  # image translation (+/- fraction)
#       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)
print(hyp)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v




def train(hyp):
    epochs = opt.epochs  # 300
    batch_size = opt.batch_size  # 64
    #weights = opt.weights  # initial training weights
    random.seed(42)
    np.random.seed(42)
    torch_utils.init_seeds(42)
    # Configure
    
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    #train_path = data_dict['train']
    #test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Create model
    #model = Model(opt.cfg, nc=data_dict['nc']).to(device)
    config = get_efficientdet_config('tf_efficientdet_d4')
    # 根据上面的配置生成网络
    load_from_pretrained = True
    #config.num_classes = 1
    #config.image_size = opt.img_size[0]
    if(load_from_pretrained):
        model = EfficientDet(config, pretrained_backbone=False)
    # 加载预训练模型
        checkpoint = torch.load(r'./tf_efficientdet_d4-5b370b7a.pth')
        model.load_state_dict(checkpoint)
        config.num_classes = 1
        config.image_size = opt.img_size[0]
    else: # load from best,last
        config.num_classes = 1
        config.image_size = opt.img_size[0]
        model = EfficientDet(config, pretrained_backbone=False)    
        checkpoint = torch.load(r'./weights/best_DEBUG.pt')#
        model.load_state_dict(checkpoint['model'].model.state_dict())
    # norm_kwargs 设置的是 BATCHNORM2D 的参数
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    model = DetBenchTrain(model, config)
    print("effDet config:",config)
    # Image sizes
    #gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [x for x in opt.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg0.append(v)
                #pg2.append(v)  # biases
                #print("bias:",k)
            elif ('.weight' in k or '.edge_weights' in k) and '.bn' not in k:
                pg1.append(v)  # apply weight decay
                #print("weight:",k)
            else:
                pg0.append(v)  # all else
                #print("else:",k)

    optimizer = optim.Adam(pg0, lr=hyp['lr0']) if opt.adam else \
        optim.RMSprop(pg0, lr=hyp['lr0'])#, momentum=hyp['momentum'])#, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
#    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Load Model

    start_epoch, best_fitness = 0, 1000.0
    '''
    if weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if model.state_dict()[k].shape == v.shape}  # to FP32, filter
            #print(ckpt['model'].keys())
            #ckpt['model'].pop('model.27.anchors') 
            #ckpt['model'].pop('model.27.anchor_grid')
            
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (opt.weights, opt.cfg, opt.weights, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt
        '''

    # Mixed precision training https://github.com/NVIDIA/apex
    model.to(device)
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)


    scheduler.last_epoch = start_epoch - 1  # do not move
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Initialize distributed training
    distribution = False
    '''
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        dist.init_process_group(backend='nccl',  # distributed backend
                                init_method='tcp://127.0.0.1:9999',  # init method
                                world_size=1,  # number of nodes
                                rank=0)  # node rank
        model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
        diftribution = True
        # pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
#'''
    # Trainloader
    dataloader = torch.utils.data.DataLoader(
                                                train_dataset,
                                                batch_size=batch_size,
                                                sampler=RandomSampler(train_dataset),
                                                pin_memory=True,#opt.cache_images,
                                                drop_last=True,
                                                num_workers=6,
                                                collate_fn=collate_fn,
                                            )
    #create_dataloader(train_path, imgsz, batch_size, gs, opt,
                 #                           hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect)
    #mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    #assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, opt.cfg)

    # Testloader
    testloader = torch.utils.data.DataLoader(
                                                validation_dataset,
                                                batch_size=batch_size,
                                                num_workers=6,
                                                shuffle=False,
                                                sampler=SequentialSampler(validation_dataset),
                                                pin_memory=True,#opt.cache_images,
                                                collate_fn=collate_fn,
                                            )
    #create_dataloader(test_path, imgsz_test, batch_size, gs, opt,
                  #                 hyp=hyp, augment=False, cache=opt.cache_images, rect=True)[0]
    

    # Exponential moving average
    ema = torch_utils.ModelEMA(model)
    #print("!!!!!!!!!!!!!!!!!! type model")
    #print(type(model))
    # Start training
    t0 = time.time()
    nb = len(dataloader)#//4  # number of batches
    n_burn = max(2 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % epochs)
    anchor = Anchor_config(config)
    anchor.anchors.to(device)
    anchor.anchor_labeler.to(device)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        
        mloss = torch.zeros(3, device="cpu")  # mean losses
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'cls', 'total', 'targets', 'img_size'))
        #ss = ('\n' + '%5d' * 7)%(0,0,0,0,0,0,0)
        
        pbar = tqdm(enumerate(dataloader),ncols= 180,total=nb)  # progress bar
        for i, (images, targets, image_ids) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            #imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            images = torch.stack(images)
            images = images.to(device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(device).float() for target in targets]# yxyx?
            labels = [target['labels'].to(device).float() for target in targets]
            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.0 + gs) // gs * gs  # size
                sf = sz / max(images.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in images.shape[2:]]  # new shape (stretched to gs-multiple)
                    images = F.interpolate(images, size=ns, mode='bilinear', align_corners=False)
            
                
            # Forward
            #cls_out,box_out = model(images)
            
            #total_loss, cls_loss, box_loss   = comp_loss_v1(cls_out,box_out,boxes,labels,config,anchor)
            total_loss, cls_loss, box_loss = model(images, boxes, labels)
            total_loss = torch.mean(total_loss)
            cls_loss = torch.mean(cls_loss)
            box_loss = torch.mean(box_loss)
            if not torch.isfinite(total_loss):
                print('WARNING: non-finite loss, ending training ', cls_loss, box_loss)
                return results

            # Backward
            if mixed_precision:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            
            # Optimize
            if ni % accumulate == 0:
                
                optimizer.step()
                optimizer.zero_grad()
                
                ema.update(model)
                
            
                
	    # Print
            
            mloss = (mloss * i + torch.tensor([box_loss*50.0,cls_loss, total_loss]).detach()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 5) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, boxes[0].shape[0], images.shape[-1])
            pbar.set_description(s)
            
            if ni < 3:
                f = 'train_batch%g.jpg' % ni  # filename
                result = plot_images(images=images, targets=boxes, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
 
            # end batch ------------------------------------------------------------------------------------------------
        
        # Scheduler
        scheduler.step()
        
        # mAP
        #ema.update_attr(model, include=['md', 'nc','gr', 'hyp', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            result = validation(model=ema.ema,val_loader = testloader,config=config,device=device)
            '''
            results, maps, times = test.test(opt.data,
                                             batch_size=batch_size,
                                             imgsz=imgsz_test,
                                             save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader)
            '''
            print("val:",result.avg)
        
        # Write
        with open(results_file, 'a') as f:
            f.write(f'[RESULT]:Train loss:{total_loss:.5f} Val. Epoch: {epoch}, summary_loss: {result.avg:.5f} \n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        #if len(opt.name) and opt.bucket:
        #    os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        

        # Update best mAP
        fi = result.avg#fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi < best_fitness:
            best_fitness = fi
            print("best_fit,\n")
        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            #with open(results_file, 'r') as f:  # create checkpoint
            '''
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': f.read(),
                    'model': ema.ema,
                    'optimizer': None if final_epoch else optimizer.state_dict()}
            '''
            ckpt = {
                        'model':ema.ema,
                        #'model_state_dict': ema.ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_summary_loss': best_fitness,
                        'epoch': epoch ,
                    }

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
    fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
    for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer
            os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    # Finish
    #if not opt.evolve:
    #    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if distribution and device.type != 'cpu' and torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    #check_git_status()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    opt.weights = last if opt.resume and not opt.weights else opt.weights
#    opt.cfg = check_file(opt.cfg)  # check file
#    opt.data = check_file(opt.data)  # check file
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # Train
    if not opt.evolve:
        tb_writer = SummaryWriter(comment=opt.name)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        train(hyp)

    
