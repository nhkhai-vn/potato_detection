#Data dir should be configured so that it contains directory: 'fried_potato'

data_dir = './data'

#Parameter for data generation
classes = ['light', 'light_defect', 'orange', 'orange_defect']
splits = {'train': 0.9, 'val': 0.1}
image_count = 10000
patches_mean = 40
patches_std = 10

#Parameter for training Yolo model
yolo_params = dict(epochs=25, batch=16, save_period=5, imgsz=(800, 800), mosaic=0, hsv_s=0.1, hsv_v=0.1, scale=0.1)

#Parameter for training Mono model
mono_params = dict(epochs=10, batch=4)
