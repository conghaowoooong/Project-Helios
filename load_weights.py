'''
Author: Conghao Wong
Date: 2021-04-06 16:55:31
LastEditors: Conghao Wong
LastEditTime: 2021-04-07 10:35:45
Description: file content
'''

import os
import tensorflow as tf
import numpy as np


def load_ckpt(checkpoint_path):
    results = {}
    variables = tf.train.list_variables(os.path.join(checkpoint_path, 'save'))

    names = []
    for item in variables:
        full_name, shape = item
        if full_name.startswith('_'):
            continue
        
        full_name_new = full_name.replace('/', '|||')
        value = tf.train.load_variable(os.path.join(checkpoint_path, 'save'), full_name)
        np.save(os.path.join(checkpoint_path, '{}.npy'.format(full_name_new)), value)
        names.append(full_name_new + '\n')

    with open(os.path.join(checkpoint_path, 'names.txt'), 'w+') as f:
        f.writelines(names)


if __name__ == '__main__':
    load_ckpt("./logs/sa_K8_zara1")