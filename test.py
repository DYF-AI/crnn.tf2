import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import json
import time
import numpy as np

from model.crnn import model
from utils.loss import CTCLoss
from utils.accuracy import WordAccuracy
from config.load_yaml import load_config
from utils.data_prepare import load_and_preprocess_image, decode_label, get_image_path, \
     load_and_preprocess_image_pridict, load_and_preprocess_image_draw

import matplotlib.pyplot as plt
from model.decode import Decoder
from matplotlib.font_manager import FontProperties

config = load_config()
print("config:", config)

train_all_image_paths, train_all_image_labels,val_all_image_paths, val_all_image_labels = get_image_path(config["work_space"]+'dataset/train/')
print(len(train_all_image_paths),len(train_all_image_labels),len(val_all_image_paths),len(val_all_image_labels))

def make_model():
    model = tf.keras.models.load_model(config["work_space"] + 'save_model/crnn_20.h5', compile=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss=CTCLoss(), metrics=[WordAccuracy()])
    return model

def make_testdataset():

    test_all_image_paths = [config["path"]["test_path"] + img for img in sorted(os.listdir(config["path"]["test_path"]))]

    test_images_num = len(test_all_image_paths)
    test_steps_per_epoch = test_images_num
    test_ds = tf.data.Dataset.from_tensor_slices(test_all_image_paths)
    test_ds = test_ds.map(
        load_and_preprocess_image_pridict, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.repeat()
    test_ds = test_ds.batch(20)
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_data = next(iter(test_ds))
    return test_data, test_all_image_paths

def test():
    test_data, test_all_image_paths = make_testdataset()
    model = make_model()
    start = time.time()
    result = model.predict(test_data)
    print('result:', result)
    print(time.time()-start)

    # with open(TABLE_PATH, 'r',encoding='gbk') as f:
    with open(config["path"]["table_path"], 'r',encoding='utf-8') as f:
        inv_table = [char.strip() for char in f]+[' ']*2
        
    decoder = Decoder(inv_table)

    inv_table

    font = FontProperties(fname='dataset/fonts/msyh.ttc', size=16)

    y_pred = decoder.decode(result, method='greedy')
    print("y_pred:", y_pred)

    for i,sentense in enumerate(y_pred):
        print('---------------------------------------')
        plt.figure(figsize=(16,32))
        plt.subplot(len(y_pred),1,i+1)
        plt.imshow(load_and_preprocess_image_draw(test_all_image_paths[i]))
        plt.xlabel('识别结果： '+sentense, fontproperties=font)
        print("sentence:", sentense)
        plt.show()

    #若要部署tensorflow serving应用请将is_save_model设置为True，生成所需文件
    #is_save_model = True
    if config["is_save_model"]:
        tf.keras.models.save_model(
                                    model,
                                    config["save_model_path"],
                                    overwrite=True,
                                    include_optimizer=True,
                                    save_format=None,
                                    signatures=None,
                                    options=None
                                )
        print('模型保存成功。')

if __name__ == '__main__':
    import fire
    fire.Fire()
