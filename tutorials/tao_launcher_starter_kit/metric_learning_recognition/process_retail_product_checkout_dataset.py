"""
Converts Retail Product Checkout (https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset) dataset to classification dataset. Ready for MLRecogNet training.
"""


import os, zipfile
import glob
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import shutil

def create_task_sets(dataset_dir, output_dir):
    # each class has 120 images in train set
    os.makedirs(output_dir, exist_ok=True)
    classes = os.listdir(dataset_dir)
    print(f"Creating subsets set from {dataset_dir}...")
    for class_name in tqdm(classes):
        samples = os.listdir(os.path.join(dataset_dir, class_name))
        sample_len = len(samples)
        indices = np.arange(sample_len)
        np.random.shuffle(indices)
        for subset_name in ["val", "test", "reference"]:
            subset_class_dir = os.path.join(output_dir, subset_name, class_name)
            if not os.path.exists(subset_class_dir):
                os.makedirs(subset_class_dir)
            
            sample_indices = np.array([])
            if subset_name == "reference":
                sample_indices = indices[:sample_len//3]
            elif subset_name == "val":
                sample_indices = indices[sample_len//3:sample_len//2]
            elif subset_name == "test":
                sample_indices = indices[sample_len//2:2*sample_len//3]
            

            final_samples = np.array(samples)[sample_indices]
            
            for sample in final_samples:
                os.rename(os.path.join(dataset_dir, class_name, sample), os.path.join(subset_class_dir, sample))
    
    print("Done!")



def crop_images(file_path, bbox, class_id, output_dir):
        
    file_name = os.path.basename(file_path)

    class_folder = os.path.join(output_dir, class_id)
    if not os.path.exists(class_folder):
        os.mkdir(class_folder)
    
    image_count = len(glob.glob( os.path.join(class_folder, file_name+"*.jpg")))
    new_file_name = os.path.join(class_folder, file_name + f"_{image_count+1}.jpg")
    if os.path.exists(new_file_name):
        # skip if file already exists
        return
    
    # start processing image
    x1, y1, x2, y2 = bbox
    
    # skip if bbox is too small
    if x2 < 120 or y2 < 150:
        return
    try:
        image = cv2.imread(file_path)
        h, w, _ = image.shape
    except:
        print(f"{file_path} is not a valid image file")
        return
    
    # give 14% margin to the bounding box
    cropped_image = image[max(int(y1 - 0.07*y2), 0 ):min(int(y1+1.07*y2), h), \
        max(int(x1 - 0.07*x2), 0 ):min(int(x1+1.07*x2), w)]

    # resize to 256x256 for faster processing and training
    resized_cropped_image = cv2.resize(cropped_image, (256, 256), cv2.INTER_AREA)
    

    cv2.imwrite(os.path.join(class_folder, new_file_name), resized_cropped_image)
        

# load dataset
data_root_dir = os.environ['HOST_DATA_DIR']
path_to_zip_file = os.path.join(data_root_dir,"retail-product-checkout-dataset.zip")
directory_to_extract_to = os.path.join(data_root_dir, "retail-product-checkout-dataset")
folder_to_extract = "retail_product_checkout"  # only extracts one folder from the zip file
processed_classification_dir = os.path.join(data_root_dir,"retail-product-checkout-dataset_classification_demo")

## unzip dataset
if not os.path.exists(processed_classification_dir):
    os.makedirs(processed_classification_dir)

print("Unzipping the dataset...")
# Ensure the folder path ends with a slash
if not folder_to_extract.endswith('/'):
    folder_to_extract += '/'

# Open the zip file
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    # Iterate over the files in the zip file
    for file_info in zip_ref.infolist():
        # Check if the file is in the specified folder
        if file_info.filename.startswith(folder_to_extract):
            # Extract the file
            zip_ref.extract(file_info, directory_to_extract_to)

directory_to_extract_to = os.path.join(directory_to_extract_to, folder_to_extract)

for dataset in ["train"]:
    dataset_dir = os.path.join(directory_to_extract_to, dataset+"2019")
    annotation_file = os.path.join(directory_to_extract_to, "instances_"+dataset+"2019.json")
    output_dir = os.path.join(processed_classification_dir, dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ## load coco dataset
    print(f"Loading COCO {dataset} dataset...")
    coco_label = COCO(annotation_file)

    # crop images to classification data
    print(f"Cropping {dataset} dataset...")
    for img_object in tqdm(coco_label.dataset["images"]):
        image_path = os.path.join(dataset_dir, img_object["file_name"])
        
        # remove top view images
        if "camera2" in image_path:
            continue
        image_id = img_object["id"]
        annotation_ids = coco_label.getAnnIds(imgIds=image_id)
        for annot in coco_label.loadAnns(annotation_ids):
            bbox = annot["bbox"]
            class_id = annot["category_id"]
            category = coco_label.loadCats(class_id)[0]
            class_name = category["supercategory"] + "_" + category["name"]
            crop_images(image_path, bbox, class_name, output_dir)

# extract a reference set from training set

# fixed random seed for reproducibility
np.random.seed(0) 
create_task_sets(
    os.path.join(processed_classification_dir, "train"), 
    os.path.join(processed_classification_dir))

# split out unknown classes
print("Select 20% of classes as unknown classes...")
class_list = os.listdir(os.path.join(processed_classification_dir, "train"))
total_class_num = len(class_list)
unknown_classes = np.random.choice(class_list, int(total_class_num*0.2), replace=False)
known_classes = [c for c in class_list if c not in unknown_classes]

known_classes_dir = os.path.join(processed_classification_dir, "known_classes")
unknown_classes_dir = os.path.join(processed_classification_dir, "unknown_classes")

for dataset in ["train", "val", "test", "reference"]:
    known_classes_dataset_dir = os.path.join(known_classes_dir, dataset)
    unknown_classes_dataset_dir = os.path.join(unknown_classes_dir, dataset)
    if not os.path.exists(known_classes_dataset_dir):
        os.makedirs(known_classes_dataset_dir)
    if not os.path.exists(unknown_classes_dataset_dir):
        os.makedirs(unknown_classes_dataset_dir)
    for class_name in tqdm(known_classes):
        class_dir = os.path.join(processed_classification_dir, dataset, class_name)
        os.rename(class_dir, os.path.join(known_classes_dataset_dir, class_name))
    for class_name in tqdm(unknown_classes):
        class_dir = os.path.join(processed_classification_dir, dataset, class_name)
        os.rename(class_dir, os.path.join(unknown_classes_dataset_dir, class_name))

# remove old folders
for dataset in ["train", "val", "test", "reference"]:
    shutil.rmtree(os.path.join(processed_classification_dir, dataset))
