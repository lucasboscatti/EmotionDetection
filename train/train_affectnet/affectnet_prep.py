import glob
import os.path

import numpy as np
import pandas as pd

from tqdm import tqdm


def create_Aff_csv():
    prefix = '/home/nero-ia/Documents/Boscatti/EmotionDetection/train/datasets/affectnet/csv'
    split = ['test_set','train_set', 'val_set']
    for set in split:
        img_path = os.path.join(prefix, set, 'images')
        ano_path = os.path.join(prefix, set, 'annotations')
        save_file = os.path.join(prefix, f'{set}_data.csv')
        images = glob.glob(img_path+'/*')
        df = pd.DataFrame(images, columns=['image_id'])
        aro = []
        val = []
        exp = []
        # lnd = []
        for image in tqdm(images, total = len(images)):
            image_name = image.split('/')[-1].split('.')[0]
            aro_file = os.path.join(ano_path, f'{image_name}_aro.npy')
            val_file = os.path.join(ano_path, f'{image_name}_val.npy')
            exp_file = os.path.join(ano_path, f'{image_name}_exp.npy')
            # lnd_file = os.path.join(ano_path, f'{image_name}_lnd.npy')

            aro.append(float(np.load(aro_file).item()))
            val.append(float(np.load(val_file).item()))
            exp.append(int(np.load(exp_file).item()))
            # lnd.append(np.load(lnd_file))
        df["labels_aro"] = aro
        df["labels_val"] = val
        df["labels_exp"] = exp
        # df["facial_landmarks"] = lnd

        df.to_csv(save_file)

if __name__ == '__main__':
    create_Aff_csv()