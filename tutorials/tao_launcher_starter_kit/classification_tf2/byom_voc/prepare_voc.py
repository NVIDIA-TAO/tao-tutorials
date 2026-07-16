import os
from os.path import join as join_path
import re
import glob
import shutil
from random import shuffle
from tqdm import tqdm


DATA_DIR=os.environ.get('LOCAL_DATA_DIR')
source_dir_orig = join_path(DATA_DIR, "VOCdevkit/VOC2012")
target_dir_orig = join_path(DATA_DIR, "formatted")

suffix = '_trainval.txt'
classes_dir = join_path(source_dir_orig, "ImageSets", "Main")
images_dir = join_path(source_dir_orig, "JPEGImages")
classes_files = glob.glob(classes_dir+"/*"+suffix)
for file in classes_files:
    # get the filename and make output class folder
    classname = os.path.basename(file)
    if classname.endswith(suffix):
        classname = classname[:-len(suffix)]
        target_dir_path = join_path(target_dir_orig, classname)
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
    else:
        continue

    with open(file) as f:
        content = f.readlines()

    for line in content:
        tokens = re.split('\s+', line)
        if tokens[1] == '1':
            # copy this image into target dir_path
            target_file_path = join_path(target_dir_path, tokens[0] + '.jpg')
            src_file_path = join_path(images_dir, tokens[0] + '.jpg')
            shutil.copyfile(src_file_path, target_file_path)

SOURCE_DIR=os.path.join(DATA_DIR, 'formatted')
TARGET_DIR=os.path.join(DATA_DIR,'split')
# list dir
dir_list = next(os.walk(SOURCE_DIR))[1]
# for each dir, create a new dir in split
for dir_i in tqdm(dir_list):
    newdir_train = os.path.join(TARGET_DIR, 'train', dir_i)
    newdir_val = os.path.join(TARGET_DIR, 'val', dir_i)
    newdir_test = os.path.join(TARGET_DIR, 'test', dir_i)
    
    if not os.path.exists(newdir_train):
        os.makedirs(newdir_train)
    if not os.path.exists(newdir_val):
        os.makedirs(newdir_val)
    if not os.path.exists(newdir_test):
        os.makedirs(newdir_test)

    img_list = glob.glob(os.path.join(SOURCE_DIR, dir_i, '*.jpg'))
    # shuffle data
    shuffle(img_list)

    for j in range(int(len(img_list) * 0.7)):
        shutil.copyfile(img_list[j], os.path.join(TARGET_DIR, 'train', os.path.join(dir_i, os.path.basename(img_list[j]))))

    for j in range(int(len(img_list) * 0.7), int(len(img_list)*0.8)):
        shutil.copyfile(img_list[j], os.path.join(TARGET_DIR, 'val', os.path.join(dir_i, os.path.basename(img_list[j]))))
            
    for j in range(int(len(img_list) * 0.8), len(img_list)):
        shutil.copyfile(img_list[j], os.path.join(TARGET_DIR, 'test', os.path.join(dir_i, os.path.basename(img_list[j]))))
                
print('Done splitting dataset.')