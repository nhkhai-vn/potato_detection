'''
    Segmentation, Annotation, Data generation 
'''


import cv2 
import numpy as np 
from scipy import ndimage 
from collections import defaultdict 
from multiprocessing import Pool 
from tqdm import tqdm 
from glob import glob 
import shutil
import os 
from yolo_export import to_yolo

from config import (
    data_dir, classes, splits, 
    image_count, patches_mean, patches_std
)

def segmentation(image, background=None, **kwargs):
    '''
        `background` should be provided for better performance on `defect` cases
    '''
    if background is None: 
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 2]
        _, mask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=5) 
    else:
        img = cv2.subtract(background, image)[:, :, 0]
        _, mask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=5) 
        mask = 1 - mask 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=5)         
    
    #Find instances and return patches
    count, markers = cv2.connectedComponents(mask, connectivity=8)
    patches = []
    for i in range(1, count):
        x, y = np.where(markers==i)
        bbox = min(x), min(y), max(x), max(y)
        inst_mask = markers[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
        img = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :].copy()
        img[inst_mask!=i, :] = 0
        inst_mask[inst_mask!=i] = 0
        inst_mask[inst_mask==i] = 1
        inst_mask = np.uint8(inst_mask)
        area = inst_mask.sum()
        inst = dict(bbox=bbox, image=img, mask=inst_mask, area=area)
        inst.update(**kwargs)
        patches.append(inst)    
    return patches, markers

    
def get_patches_from_files(files, background_files):
    '''
        Perform segmentation for all given `files` 
        Extract all patches and select correct patches using Area measurement 
    '''
    backgrounds = [cv2.imread(x) for x in background_files] 
    
    #All patches from all files 
    patches = []
    for file in tqdm(files, 'extracting patches'): 
        image = cv2.imread(file) 
        label, file_name = file.split('/')[-2:] 
        instances, markers = segmentation(
            image, 
            backgrounds[0] if 'defect' in label else None, 
            origin=file_name, 
            label=label
        ) 
        patches.extend(instances)
    
    #Filtering 
    areas = [x['area'] for x in patches]
    min_area = np.quantile(areas, 0.10)
    max_area = np.quantile(areas, 0.80)
    patches = [x for x in patches if x['area'] >= min_area and x['area'] <= max_area]        
    
    #Merge patches by class and return
    patches_by_class = defaultdict(lambda: [])
    for patch in patches:
        patches_by_class[patch['label']].append(patch)
    patches = list(patches_by_class.values())
    return patches, backgrounds 


def rotate(mats, p=0.5, max_angle=90):
    if np.random.rand()>=p: 
        angle = np.random.randint(max_angle)
        return [np.uint8(ndimage.rotate(m, angle)) for m in mats]
    return mats 
    
 
def create_image(patches, backgrounds, patches_mean, patches_std):    
    '''
        Create synthesized image given patches, backgrounds, and near-expected number of instances 
    '''
    background = backgrounds[np.random.randint(len(backgrounds))] #chose random background uniformly 
    instance_count = max(5, int(np.random.normal(patches_mean, patches_std))) #choose number of patches normally
    image_height, image_width = background.shape[:2] 
    image = np.copy(background) 
    depth_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    inst_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    raw_instances = []
    for i in range(instance_count):
        ps = patches[np.random.randint(len(patches))]
        patch = ps[np.random.randint(len(ps))]
        
        patch_image, patch_mask = rotate((patch['image'], patch['mask']), p=0.5) #random rotate 
        patch_mask = cv2.erode(patch_mask, np.ones((5, 5), np.uint8), iterations=1) #ensure foreground after rotation 
        patch_mask = patch_mask!=0 
        area = patch_mask.sum()
        if area == 0: 
            continue 
        
        #put on to image         
        patch_height, patch_width = patch_mask.shape 
        h, w = image_height - patch_height, image_width - patch_width 
        top = min(max(int(np.random.normal(h/2, h/6)), 0), h) #choose location, near to center 
        left = min(max(int(np.random.normal(w/2, w/6)), 0), w) #choose location, near to center  

        #put object on image 
        bbox = image[top: top + patch_height, left:left + patch_width, :]
        bbox[patch_mask, :] = patch_image[patch_mask, :] 
        
        #depth mask
        depth = depth_mask[top: top + patch_height, left:left + patch_width]
        depth[patch_mask] = depth[patch_mask].max() + 1 
        
        #instance mask 
        mask = inst_mask[top: top + patch_height, left:left + patch_width]
        mask[patch_mask] = i + 1
        
        #instances 
        bbox = [left, top, patch_width, patch_height]
        raw_instances.append(dict(label=patch['label'], bbox=bbox, mask=mask, area=area)) 
    
    #refined overlaped mask 
    instances = []
    for i,inst in enumerate(raw_instances):
        mask = inst['mask'].copy()
        mask[mask!=i+1] = 0 
        area = mask.sum()
        if area / inst['area'] >= 0.25: 
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,  offset=(-1, -1)) 
            left_top = np.array(inst['bbox'][:2])
            contours = [(x + left_top) for x in contours if x.size>=6]
            inst['segmentation'] = [x.flatten() for x in contours]
            inst['area'] = area 
            inst['mask'] = mask
            instances.append(inst) 
    return image, instances, depth_mask


def export(data, data_dir, splits_list, classes, imwidth, imheight):    
    '''
        Export annotation to Yolo format 
    '''
    #Label files 
    class_map = {x:i for i, x in enumerate(classes)}
    for i in tqdm(range(len(data)), 'export'):
        image_path = f'{data_dir}/images/{splits_list[i]}/{i + 1}.png'
        label_path = f'{data_dir}/labels/{splits_list[i]}/{i + 1}.txt'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        shutil.move(f'{data_dir}/images/{i + 1}.png', image_path)
        yolos = [] 
        for inst in data[i]:
            pts = to_yolo(inst['segmentation'], imwidth, imheight) 
            pts = [class_map[inst['label']]] + pts
            yolos.append(' '.join([str(x) for x in pts]) + '\n') 
        with open(label_path, 'wt') as fout: 
            fout.writelines(yolos) 
            
    #YOLO yaml file 
    with open(data_dir + '/dataset.yaml', 'wt') as fout:
        fout.write(
            'nc: {}\n'\
            'names: {}\n'\
            'path: {}\n'
            'train: images/train\n'\
            'val: images/val\n'\
            .format(len(classes), str(list(classes)), os.path.abspath(data_dir))
        )
    
    
    #Depth split files 
    depth_splits = [[], []]
    for i, split in enumerate(splits_list):
        s = f'{data_dir}/images/{split}/{i + 1}.png {data_dir}/depth/{i + 1}.png 50 \n' 
        depth_splits[int(split=='val')].append(s)
    with open(data_dir + '/depth_train.txt', 'wt') as fout:
        fout.writelines(depth_splits[0]) 
    with open(data_dir + '/depth_val.txt', 'wt') as fout:
        fout.writelines(depth_splits[1] )


_worker = {}


def _init_worker(patches, backgrounds):
    _worker.update(patches=patches, backgrounds=backgrounds)


def _generate_image(index):
    np.random.seed(index)
    image, instances, depth_mask = create_image(_worker['patches'], _worker['backgrounds'], patches_mean, patches_std)
    cv2.imwrite(f'{data_dir}/images/{index}.png', image)
    cv2.imwrite(f'{data_dir}/depth/{index}.png', depth_mask)
    return instances


def generate_data():        
    '''
        Wrap above functions to perform segmenation and data generation 
        Parameters are controlled by config.py 
    '''
    input_dir = f'{data_dir}/fried_potato' 
    
    background_files = sorted(glob(f'{input_dir}/background/*.png'))
    files = sum([glob(f'{input_dir}/{x}/*.png') for x in classes], [])
    patches, backgrounds = get_patches_from_files(files, background_files)
    
    imheight, imwidth = backgrounds[0].shape[:2]
    for ps in patches:
        for i, patch in enumerate(ps):
            os.makedirs(f'{data_dir}/patches/{patch["label"]}', exist_ok=True)
            cv2.imwrite(f'{data_dir}/patches/{patch["label"]}/{i}.png', patch['image'])
    
    os.makedirs(f'{data_dir}/images', exist_ok=True)
    os.makedirs(f'{data_dir}/depth', exist_ok=True)
    with Pool(16, initializer=_init_worker, initargs=(patches, backgrounds)) as pool:
        data = list(tqdm(pool.imap(_generate_image, range(1, image_count + 1)), 'augmentation', total=image_count))
    splits_list = sum([[k] * int(v * image_count) for k, v in splits.items()], [])
    export(data, data_dir, splits_list, classes, imwidth, imheight)
    
    
if __name__=='__main__': 
    generate_data()
