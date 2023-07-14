'''
    Unet model with DenseNet backbone     
    Original 
        https://gist.github.com/bmanikan/1cf05d6cc8a02d07ae66da7bdd003c9e#file-depth_model-py
        https://gist.github.com/bmanikan/b2721e89e9b009a33fb7f327fff51473#file-depth_datagenerator-py
        https://gist.github.com/bmanikan/0086d5e3313acd8a01deaa1bdb611539#file-depth_trainingpipeline-py

    Modified by Khai 
'''
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, concatenate 
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import numpy as np 
import cv2 

    
def upsampling(input_tensor, n_filters, concat_layer):
    # Bilinear 2x upsampling layer
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(input_tensor)
    x = concatenate([x,concat_layer])
    x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    return x


def build_model(dim=(640, 480)):
    names = ['pool3_pool', 'pool2_pool', 'pool1','conv1/relu']
    encoder = DenseNet169(include_top=False, weights='imagenet', input_shape=(dim[1], dim[0], 3))
    for layer in encoder.layers:
        layer.trainable = True
    inputs = encoder.input
    x = encoder.output
    bneck = Conv2D(filters=1664, kernel_size=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(bneck, 832, encoder.get_layer(names[0]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 416, encoder.get_layer(names[1]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 208, encoder.get_layer(names[2]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 104, encoder.get_layer(names[3]).output)
    x = Conv2D(filters=1, activation='sigmoid', kernel_size=(3,3), padding='same')(x)
    model = Model(inputs=inputs, outputs=x)
    return model 


def loss_function(y_true, y_pred):
    #Cosine distance loss
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)    
    # edge loss for sharp edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    # structural similarity loss
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)
    
    w1, w2, w3 = 1.0, 1.0, 0.1
    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


def accuracy_function(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def polynomial_decay(epoch):
    # Learning rate scheduler
    max_epochs = 10
    base_lr = 0.0001
    power = 1.0
    lr = base_lr * (1 - (epoch / float(max_epochs))) ** power
    return lr


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, dim, depth_dim, shuffle=False):
        self.df = dataframe
        self.width, self.height = dim
        self.depth_width, self.depth_height = depth_dim
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ids = dataframe['image'].unique()
        self.imgtodpth = dataframe.set_index('image')['depth'].to_dict()
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
              np.random.shuffle(self.indexes)
    
    def __getitem__(self,index):
        idxs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        flip = np.random.choice([True, False])
        query_imgs = [self.ids[idx] for idx in idxs]
        target_imgs = [self.imgtodpth[img] for img in query_imgs]
        processed_query_img = self._preprocess_image(query_imgs, flip)
        processed_depth_img = self._preprocess_depth(target_imgs, flip)
        return processed_query_img, processed_depth_img


    def _preprocess_image(self, images, flip):
        '''
            Resize, Normalize and randomly Augments the image set. 
        '''
        processed = []
        for img in images:
            resized_img = cv2.resize(cv2.imread(img),(self.width, self.height)).astype(np.float32)
            scaled_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
            if flip:
                scaled_img = cv2.flip(scaled_img, 1)
            processed.append(scaled_img)
        return np.array(processed)


    def _preprocess_depth(self, images, flip):
        '''
            Resize, Normalize and randomly Augments the depth maps.
        '''
        processed = []
        for img in images:
            img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(img, (self.depth_width, self.depth_height)).astype(np.float32)
            scaled_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
            if flip:
                scaled_img = cv2.flip(scaled_img, 1)
            scaled_img = np.expand_dims(scaled_img, axis=-1)
            processed.append(scaled_img)
        return np.array(processed)

