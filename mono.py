from dense_unet import *
import tensorflow as tf
from collections import namedtuple
import pandas as pd 
import numpy as np 
import cv2 

        
class MONO():
    def __init__(self, model_path=None):
        if model_path:
            self.model = build_model()
            self.model.load_weights(model_path)
        self.dim = (640, 480)
        self.depth_dim = (320, 240)
        #TODO: provide flexible customization 

    def train(self, data, **kwargs):
        args = dict(
            epochs=10,
            batch=8, 
            learning_rate=0.0001,
            amsgrad=True,
            save_best_only=True,
            verbose=1
        )
        args.update(kwargs)
        args = namedtuple('args', args)(**args)
        
        train_data, val_data = data 
        train_data = pd.read_csv(train_data, header=None, sep='\s+', names=['image', 'depth', 'fl'])
        val_data = pd.read_csv(val_data, header=None, sep='\s+', names=['image', 'depth', 'fl'])
        train_generator = DataGenerator(train_data, batch_size=args.batch, dim=self.dim, depth_dim=self.depth_dim, shuffle=True)
        val_generator = DataGenerator(val_data, batch_size=args.batch, dim=self.dim, depth_dim=self.depth_dim, shuffle=False)
        
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=args.amsgrad)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('runs/depth/cp.ckpt', save_weights_only=True, save_best_only=args.save_best_only, verbose=args.verbose)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='runs/depth/', histogram_freq=0, write_graph=True, write_images=True, update_freq=10)
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(polynomial_decay, verbose=1)
        callbacks = [scheduler_callback, checkpoint_callback, tensorboard_callback]
        self.model = model = build_model()
        model.compile(optimizer=opt, loss=loss_function, metrics=[accuracy_function]) 
        model.summary()
        
        model.fit(train_generator, validation_data=val_generator, epochs=args.epochs, callbacks=callbacks)
        
        
    def _preprocess(self, img):
        img = np.array(img)[:,:,::-1] 
        img = cv2.resize(img, self.dim) 
        img = (img - img.min()) / (img.max() - img.min())
        return img 
    

    def __call__(self, images, segments): 
        images = np.array([self._preprocess(x) for x in images])
        depths = self.model.predict(images) 
        depths = np.squeeze(depths, axis=-1)        
        
        local_depths = []
        for seg, dep in zip(segments, depths):
            if len(seg)==0:
                local_depths.append(dep)
                continue
            dep = cv2.resize(dep, seg[0].masks.data[0].shape[::-1]) #match the size of two masks
            
             #step 1. from global depth to local depth 
            local_dep = np.zeros(dep.shape) #local depth mask 
            for patch in seg: 
                mask = patch.masks.data[0]==1 
                local_dep[mask] = dep[mask].mean() #use mean value for local depth 

            #step 2. from local depth to relative depth 
            #shrink objects to skip small overlaps 
            bin_mask = np.uint8(local_dep!=0)
            bin_mask = cv2.erode(bin_mask, np.ones((11, 11), np.uint8))
            count, markers = cv2.connectedComponents(bin_mask, connectivity=8)
            local_dep = local_dep - 0.0000001
            for i in range(1, count):
                m = markers == i
                for j, k in enumerate(np.unique(local_dep[m])):
                    n = local_dep == k
                    if (n & m).sum() >= 100:
                        local_dep[n & m] = j + 1
            local_dep[local_dep < 1] = 0
            local_dep = local_dep / local_dep.max()
            local_dep = cv2.dilate(local_dep, np.ones((11, 11), np.uint8)) #restore mask 
            local_depths.append(local_dep) 
            
        return depths, local_depths

