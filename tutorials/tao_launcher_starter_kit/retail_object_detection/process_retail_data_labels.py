"""
Convert the 2d object detection labels from 200-class format to binary class format
"""
import json, os

for dataset in ["train", "val", "test"]:
    print(f"Processing {dataset} dataset")
    input_label_file=os.path.join(os.getenv('HOST_DATA_DIR'),'retail_object_detection',f'instances_{dataset}2019.json')
    output_label_file=os.path.join(os.getenv('HOST_DATA_DIR'),'retail_object_detection',f'binary_{dataset}2019.json')

    with open(input_label_file) as f:
        data = json.load(f)

    data['categories'] = [{'id': 1, 'name': 'retail object'}]
    for ann in data['annotations']:
        ann['category_id'] = 1

    with open(output_label_file, 'w') as f:
        json.dump(data, f)

print("Done!")