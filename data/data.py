import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2,ToTensor
import random
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset,RandomSampler,SequentialSampler
import os
import math
import time
import datetime
import glob
import warnings
warnings.filterwarnings("ignore")

def delete_add(df, delete_index, add_xywh, delete=True):
    replaces = []
    for xywh in add_xywh:
        replace = df.iloc[[delete_index]].copy()
        replace[['x','y','w','h']] = xywh
        replaces.append(replace)
    replaces = pd.concat(replaces)
    if delete:
        df = df.drop(index=delete_index)
    df = pd.concat([df, replaces]).reset_index(drop=True)
    return df

def makeFold(data_path):
    #df = pd.read_csv(r'../data/train.csv')
    df = pd.read_csv(r'../data/train.csv')
    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:,i]
    df.drop(columns=['bbox'], inplace=True)
    df = delete_add(df, 173, [[0,15,140,70],[288,270,80,150]]) #"41c0123cc"
    df = delete_add(df, 2158, [[10,449,320,68],[85,530,300,80],[290,430,225,200],[640,110,110,80]]) #"a1321ca95"
    df = delete_add(df, 2167, [[137,610,270,220]]) #"a1321ca95"
    df = delete_add(df, 3684, [[37,84,90,140]]) #"2cc75e9f5"
    df = delete_add(df, 52864, [[3,470,210,180],[213,280,250,200]]) #"9a30dd802"
    df = delete_add(df, 113942, [[262,645,70,70]]) #"42e6efaaa"
    df = delete_add(df, 117338, [[492,268,100,100]]) #"409a8490c"
    df = delete_add(df, 118204, [[922,857,46,58]]) #"d7a02151d"
    df = delete_add(df, 121625,  [[0,673,55,64],[575,133,129,69]]) #"d067ac2b1"
    df = delete_add(df, 121625, [[672,37,155,107],[770,313,210,80]]) #"d067ac2b1"
    df = delete_add(df, 147494, [[325,62,94,77]]) #"d60e832a5"
    df = delete_add(df, 120729, [[890,890,133,133]], False) #"d8cae4d1b"
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_folds = df[['image_id']].copy()
    
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
    	df_folds['source'].values.astype(str),
    	df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
	)
    np.unique(df_folds.stratify_group)
    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    
    return df_folds, df
    
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit= 0.3,
                                     val_shift_limit=0.3, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                           contrast_limit=0.3, p=0.8),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.8),
            #ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

# 验证集阶段的数据增强变换
# 依托于 albumentations 这个三方包
def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            #ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )
   
    
TRAIN_ROOT_PATH = r'../data/train'
EXTERNAL_DATA = '../data/adv_data_all'

# Torch 的数据生成器 
class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None,back_up_trans = None, test=False):
        super().__init__()
        
        # 图片的 ID 列表
        self.image_ids = image_ids
        # 图片的标签和基本信息
        self.marking = marking
        # 图像增强
        self.transforms = transforms
        # 测试集
        self.test = test
        
        # Mosaic and mixuo
        self.image_external = glob.glob(EXTERNAL_DATA+'/*.jpg')
        self.mosaic_border = (0,0)
        self.img_size = 1024
        self.mixup_ratio = 0.5
        self.min_box_surf = 20 #pix*pix

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        # 百分之 50 的概率会做 mix up
        if self.test or random.random() >1 :
            # 具体定义在后面
            image, boxes = self.load_image_and_boxes(index)
        else:
#             # 具体定义在后面
# #             image, boxes = self.load_mixup_image_and_boxes(index)
#             image, boxes = self.load_mixup_image(index)
            image, boxes = self.load_mosaic_mixup(index)        
        # 这里只有一类的目标定位问题，标签数量就是 bbox 的数量
        labels = torch.ones((boxes.shape[0],), dtype=torch.int32)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        
        # 多做几次图像增强，防止有图像增强失败，如果成功，则直接返回。
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    break
            if i >9:
                print("!!!!!!!!!!panic!!!!!!A crashes")
                sample = self.back_up_trans(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning

        #print(image.shape)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        # 加载 image_id 名字
        image_id = self.image_ids[index]
        # 加载图片
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # 转换图片通道 从 BGR 到 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 0,1 归一化
        image /= 255.0
        # 获取对应 image_id 的信息
        records = self.marking[self.marking['image_id'] == image_id]
        # 获取 bbox
        boxes = records[['x', 'y', 'w', 'h']].values
        # 转换成模型输入需要的格式
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_mixup_image_and_boxes(self,index,imsize=1024):
        # 加载图片和 bbox
        image, boxes = self.load_image_and_boxes(index)
        # 随机加载另外一张图片和 bbox
        r_image, r_boxes= self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        # 进行 mixup 图片的融合，这里简单的利用 0.5 权重
        mixup_image = (image + r_image) / 2
        # 进行 mixup bbox的融合
        mixup_boxes = np.concatenate((boxes,r_boxes),0)
        return mixup_image, mixup_boxes
    
    ## Only mixup with texture image
    def load_image_external(self, index):
        # 加载 image_id 名字
        image_id = self.image_external[index]
        # 加载图片
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        # 转换图片通道 从 BGR 到 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 0,1 归一化
        image /= 255.0
        return image
    
    def load_mixup_image(self,index,imsize=1024):
        mix_alpha = 0.5
        image, boxes = self.load_image_and_boxes(index)
        indice_external = random.randint(0, len(self.image_external) - 1)
        image_external = self.load_image_external(indice_external)
        mixup_image = cv2.addWeighted(image_external, mix_alpha, image, 1-mix_alpha,0)
        
        return mixup_image, boxes

    ## Mosaic + Mixup
    def load_mosaic_mixup(self, index): ## add mixup (meng-liao)
        boxes4 = []
        s = self.img_size//2
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.image_ids) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load mixed image
            if random.uniform(0,1)<self.mixup_ratio:
                img, boxes = self.load_mixup_image(index)
            else:
                img, boxes = self.load_image_and_boxes(index)
            h, w = img.shape[:2]
            
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 0.5, dtype='float32')  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # mosaic boxes
            boxes_mosaic = boxes.copy()
            if boxes.size > 0:
                boxes_mosaic[:, 0] = boxes[:, 0] + padw
                boxes_mosaic[:, 1] = boxes[:, 1] + padh
                boxes_mosaic[:, 2] = boxes[:, 2] + padw
                boxes_mosaic[:, 3] = boxes[:, 3] + padh
            boxes4.append(boxes_mosaic)
        
        # Concat/clip labels
        if len(boxes4):
            boxes4 = np.concatenate(boxes4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            
            ## boundary crop
            np.clip(boxes4, 0, 2 * s, out=boxes4)  
            boxes4 = boxes4[(boxes4[:,2]-boxes4[:,0])*(boxes4[:,3]-boxes4[:,1])>self.min_box_surf,:]

#         plt.figure()
#         plt.imshow(img4)
#         plt.title('img4')
        return img4, boxes4    
    
def collate_fn(batch):
    return tuple(zip(*batch))    
    
    
fold_number = 0
df_folds,marking = makeFold('../data/train.csv')

train_dataset = DatasetRetriever(
    image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
    marking=marking,
    transforms=get_train_transforms(),
    back_up_trans = get_valid_transforms(),
    test=False,
)

validation_dataset = DatasetRetriever(
    image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
    marking=marking,
    transforms=get_valid_transforms(),
    test=True,
)



def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[1]), int(x[0])), (int(x[3]), int(x[2]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if 0:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    
def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=512, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        
        images = images.cpu().float().numpy()
    else:
        images = []
        for img in images:
            try:
                img = img.cpu().float().numpy
            except:
                pass
            images.append(img)
        images = np.stack(images)
        print('something wrong')
        #images = np.vsplit(images, images.shape[0])
    
    if isinstance(targets[0], torch.Tensor):
        targets = [x.cpu().numpy() for x in targets]

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    #bs = len(images)
    bs,_,h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[i]
            boxes = image_targets.T #boxes = xywh2xyxy(image_targets).T
            #classes = [0]#image_targets[:, 1].astype('int')
            gt = True#image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= scale_factor
            boxes[[0, 2]] += block_y
            boxes[[1, 3]] *= scale_factor
            boxes[[1, 3]] += block_x
            for j, box in enumerate(boxes.T):
                cls = 0#int(classes[j])
                color = color_lut[1 % len(color_lut)]
                #cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, "wheat", (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic

    
