# Define the decoding functions.
import numpy as np
import cv2
import os
import glob
import tqdm
import json
import requests
import argparse
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


OBJECTRON_BUCKET = "gs://objectron/v1/records_shuffled"
PUBLIC_URL = "https://storage.googleapis.com/objectron"
DATA_DISTRIBUTION = ['train', 'test', 'val']


def get_image(feature, shape=None):
    """Decode the tensorflow image example."""
    image = cv2.imdecode(
        np.asarray(bytearray(feature.bytes_list.value[0]), dtype=np.uint8),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if shape is not None:
        image = cv2.resize(image, shape)
    return image


def parse_plane(example):
    """Parses plane from a tensorflow example."""
    fm = example.features.feature
    if "plane/center" in fm and "plane/normal" in fm:
        center = fm["plane/center"].float_list.value
        center = np.asarray(center)
        normal = fm["plane/normal"].float_list.value
        normal = np.asarray(normal)
        return center, normal
    else:
        return None


def parse_example(example):
    """Parse the image example data"""
    fm = example.features.feature

    # Extract images, setting the input shape for Objectron Dataset
    image = get_image(fm["image/encoded"], shape=(600, 800))
    filename = fm["image/filename"].bytes_list.value[0].decode("utf-8")
    filename = filename.replace('/', '_')
    image_id = np.asarray(fm["image/id"].int64_list.value)[0]

    label = {}
    visibilities = fm["object/visibility"].float_list.value
    visibilities = np.asarray(visibilities)
    index = visibilities > 0.1

    if "point_2d" in fm:
        points_2d = fm["point_2d"].float_list.value
        points_2d = np.asarray(points_2d).reshape((-1, 9, 3))[..., :2]

    if "point_3d" in fm:
        points_3d = fm["point_3d"].float_list.value
        points_3d = np.asarray(points_3d).reshape((-1, 9, 3))

    if "object/scale" in fm:
        obj_scale = fm["object/scale"].float_list.value
        obj_scale = np.asarray(obj_scale).reshape((-1, 3))

    if "object/translation" in fm:
        obj_trans = fm["object/translation"].float_list.value
        obj_trans = np.asarray(obj_trans).reshape((-1, 3))

    if  "object/orientation" in fm:
        obj_ori = fm["object/orientation"].float_list.value
        obj_ori = np.asarray(obj_ori).reshape((-1, 3, 3))

    label["2d_instance"] = points_2d[index]
    label["3d_instance"] = points_3d[index]
    label["scale_instance"] = obj_scale[index]
    label["translation"] = obj_trans[index]
    label["orientation"] = obj_ori[index]
    label["image_id"] = image_id
    label["visibility"] = visibilities[index]
    label['ORI_INDEX'] = np.argwhere(index).flatten()
    label['ORI_NUM_INSTANCE'] = len(index)
    return image, label, filename


def parse_camera(example):
    """Parse the camera calibration data"""
    fm = example.features.feature
    if "camera/projection" in fm:
        proj = fm["camera/projection"].float_list.value
        proj = np.asarray(proj).reshape((4, 4))
    else:
        proj = None
        
    if "camera/view" in fm:
        view = fm["camera/view"].float_list.value
        view = np.asarray(view).reshape((4, 4))
    else:
        view = None
    
    if "camera/intrinsics" in fm:
        intrinsic = fm["camera/intrinsics"].float_list.value
        intrinsic = np.asarray(intrinsic).reshape((3, 3))
    else:
        intrinsic = None
    return proj, view, intrinsic


def partition(lst, n):
    """Equally split the video lists."""
    division = len(lst) / float(n) if n else len(lst)
    return [lst[int(np.round(division * i)): int(np.round(division * (i + 1)))] for i in range(n)]


def generate_data(test_categories, num_images=-1):

    save_dir = os.path.join(os.environ['DATA_DIR'])

    for c in test_categories:
        for dist in DATA_DISTRIBUTION:
            # Download the tfrecord files
            if dist in ['test', 'val']:
                eval_data = f'/{c}/{c}_test*'
                blob_path = PUBLIC_URL + f"/v1/index/{c}_annotations_test"
            elif dist in ['train']:
                eval_data = f'/{c}/{c}_train*'
                blob_path = PUBLIC_URL + f"/v1/index/{c}_annotations_train"
            else:
                raise ValueError("No specific data distribution settings.")

            eval_shards = tf.io.gfile.glob(OBJECTRON_BUCKET + eval_data)
            ds = tf.data.TFRecordDataset(eval_shards).take(num_images)

            with tf.io.TFRecordWriter(f'{save_dir}/{c}_{dist}.tfrecord') as file_writer:
                for serialized in tqdm.tqdm(ds): 
                    example = tf.train.Example.FromString(serialized.numpy())
                    record_bytes = example.SerializeToString()
                    file_writer.write(record_bytes)

            # Get the video ids
            video_ids = requests.get(blob_path).text
            video_ids = [i.replace('/', '_') for i in video_ids.split('\n')]
            
            # Work on a subset of the videos for each round, where the subset is equally split
            video_ids_split = partition(video_ids, int(np.floor(len(video_ids) / int(len(video_ids) / 2))))

            # Decode the tfrecord files
            tfdata = f'{save_dir}/{c}_{dist}*'
            eval_shards = tf.io.gfile.glob(tfdata)

            new_ds = tf.data.TFRecordDataset(eval_shards).take(-1)
            for subset in video_ids_split:
                videos = {}
                for serialized in tqdm.tqdm(new_ds):

                    example = tf.train.Example.FromString(serialized.numpy())

                    # Group according to video_id & image_id
                    fm = example.features.feature
                    filename = fm["image/filename"].bytes_list.value[0].decode("utf-8")
                    video_id = filename.replace('/', '_')
                    image_id = np.asarray(fm["image/id"].int64_list.value)[0]
                    
                    # Sometimes, data is too big to save, so we only focus on a small subset instead.
                    if video_id not in subset:
                        continue

                    if video_id in videos:
                        videos[video_id].append((image_id, example))
                    else:
                        videos[video_id] = []
                        videos[video_id].append((image_id, example))
                
                # Saved the decoded tfrecord files. 
                save_tfrecords = f'{save_dir}/{c}/tfrecords/{dist}'
                if not os.path.exists(save_tfrecords):
                    os.makedirs(save_tfrecords)
                for video_id in tqdm.tqdm(videos):
                    with tf.io.TFRecordWriter(f'{save_tfrecords}/{video_id}.tfrecord') as file_writer:
                        for image_data in videos[video_id]:
                            record_bytes = image_data[1].SerializeToString()
                            file_writer.write(record_bytes)

            # Extract the images and ground truth.
            videos = [os.path.splitext(os.path.basename(i))[0] for i in glob.glob(f'{save_tfrecords}/*.tfrecord')]
            if dist in ['train', 'val']:
                frame_rate = 15
            elif dist in ['test']:
                frame_rate = 1
            else:
                raise ValueError("No specific data distribution settings.")
            
            for idx, key in enumerate(videos):
                ds = tf.data.TFRecordDataset(f'{save_tfrecords}/{key}.tfrecord').take(-1)

                for serialized in tqdm.tqdm(ds):
                    example = tf.train.Example.FromString(serialized.numpy())

                    image, label, prefix = parse_example(example)
                    frame_id = label['image_id']

                    if int(frame_id) % frame_rate == 0:
                        
                        proj, view, cam_intrinsic = parse_camera(example)
                        plane = parse_plane(example)

                        cam_intrinsic[:2, :3] = cam_intrinsic[:2, :3] / 2.4
                        center, normal = plane
                        height, width, _ = image.shape

                        im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        dict_out = {
                            "camera_data" : {
                                "width" : width,
                                'height' : height,
                                'camera_view_matrix':view.tolist(),
                                'camera_projection_matrix':proj.tolist(),
                                'intrinsics':{
                                    'fx':cam_intrinsic[1][1],
                                    'fy':cam_intrinsic[0][0],
                                    'cx':cam_intrinsic[1][2],
                                    'cy':cam_intrinsic[0][2]
                                }
                            }, 
                            "objects" : [],
                            "AR_data":{
                                'plane_center':[center[0],
                                                center[1],
                                                center[2]],
                                'plane_normal':[normal[0],
                                                normal[1],
                                                normal[2]]
                            }
                        }
                        
                        for object_id in range(len(label['2d_instance'])):
                            object_categories = c
                            quaternion = R.from_matrix(label['orientation'][object_id]).as_quat()
                            trans = label['translation'][object_id]

                            projected_keypoints = label['2d_instance'][object_id]
                            projected_keypoints[:, 0] *= width
                            projected_keypoints[:, 1] *= height

                            object_scale = label['scale_instance'][object_id]
                            keypoints_3d = label['3d_instance'][object_id]
                            visibility = label['visibility'][object_id]

                            dict_obj={
                                'class': object_categories,
                                'name': object_categories+'_'+str(object_id),
                                'provenance': 'objectron',
                                'location': trans.tolist(),
                                'quaternion_xyzw': quaternion.tolist(),
                                'projected_cuboid': projected_keypoints.tolist(),
                                'scale': object_scale.tolist(),
                                'keypoints_3d': keypoints_3d.tolist(),
                                'visibility': visibility.tolist()
                            }
                            # Final export
                            dict_out['objects'].append(dict_obj)

                        save_path = f"{save_dir}/{c}/{dist}/{prefix}/"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        filename = f"{save_path}/{str(frame_id).zfill(5)}.json"
                        with open(filename, 'w+') as fp:
                            json.dump(dict_out, fp, indent=4, sort_keys=True)
                        cv2.imwrite(f"{save_path}/{str(frame_id).zfill(5)}.png", im_bgr)


def parse_args():
    parser = argparse.ArgumentParser("Objectron dataset for training the CenterPose.")
    parser.add_argument(
        "-c", "--categories",
        metavar='N', type=str, nargs='+', required=True,
        help="Download and preprocess the specific categories."
    )
    parser.add_argument(
        "-n", "--num_imgs",
        type=int, required=True,
        default=-1,
        help="Download number of images. Default=-1 (download whole dataset)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_data(args.categories, args.num_imgs)
