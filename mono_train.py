from mono import MONO 
from config import data_dir, mono_params 

if __name__ == '__main__':
    model = MONO()
    model.train([f'{data_dir}/depth_train.txt', f'{data_dir}/depth_val.txt'], **mono_params)
