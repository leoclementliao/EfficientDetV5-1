import os
import time
import torch

def strip_optimizer(f='weights/best.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 1/2 size)
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer_state_dict'] = None
    x['scheduler_state_dict'] = None
    x['model'].half()  # to FP16
    torch.save(x, f)
    print('Optimizer stripped from %s, %.1fMB' % (f, os.path.getsize(f) / 1E6))


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def validation(model, val_loader,config,device):
    # 切换到模型的验证模式
    model.eval()
    # 初始化损失计算器
    summary_loss = AverageMeter()
    t = time.time()
    # 开始遍历验证集
    for step, (images, targets, image_ids) in enumerate(val_loader):
        if config.verbose:
            if step % config.verbose_step == 0:
                print(
                    f'Val Step {step}/{len(val_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )
        with torch.no_grad():
            images = torch.stack(images)
            batch_size = images.shape[0]
            images = images.to(device).float()
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]

            loss, _, _ = model(images, boxes, labels)
            summary_loss.update(loss.detach().item(), batch_size)

    return summary_loss
