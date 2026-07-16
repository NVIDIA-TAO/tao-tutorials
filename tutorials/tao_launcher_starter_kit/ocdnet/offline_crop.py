import os
import glob
import cv2
import math
import numpy as np
from shapely.geometry import Polygon, mapping
import copy
import argparse


def keep_aspect_ratio_resize(ori_img, ori_w, ori_h, new_w, new_h):
    # Crop image, keep aspect ratio
    
    scale_w = new_w / ori_w
    scale_h = new_h / ori_h
    scale = min(scale_w, scale_h)
    h = int(ori_h * scale)
    w = int(ori_w * scale)
    
    if len(ori_img.shape) == 3:
        padimg = np.zeros((new_h, new_w, ori_img.shape[2]), ori_img.dtype)
    else:
        padimg = np.zeros((new_h, new_w), ori_img.dtype)
    padimg[:h, :w] = cv2.resize(ori_img, (w, h))
    img = padimg
    return img, scale


def order_points_clockwise(pts):
    """order points clockwise."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_annotation(label_path: str) -> dict:
    "get annotation"
    boxes = []
    texts = []
    ignores = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                if cv2.contourArea(box) > 0:
                    boxes.append(box)
                    label = params[8]
                    texts.append(label)
                    ignores.append(label in ['*', '###'])
            except Exception:
                print('load label failed on {}'.format(label_path))
    data = {
        'text_polys': np.array(boxes),
        'texts': texts,
        'ignore_tags': ignores,
    }
    return data


def parse_args(args=None):
    """parse the arguments."""
    parser = argparse.ArgumentParser(description='Offline crop for large resolution images')

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Input dataset directory of training images and labels."
    )
    parser.add_argument(
        "--has-gt",
        type=bool,
        required=False,
        default=True,
        help="if GT labels are needed to crop, default True. if dataset has no labels, set this to False."
    )
    parser.add_argument(
        "--img-ext",
        type=str,
        required=False,
        default='jpg',
        help="image ext. such as jpg, png, jpeg"
    )

    parser.add_argument(
        "--patch-height",
        type=int,
        required=True,
        help="height of cropped patch."
    )

    parser.add_argument(
        "--patch-width",
        type=int,
        required=True,
        help="Width of cropped patch."
    )

    parser.add_argument(
        "--overlapRate",
        type=float,
        required=False,
        default=0.5,
        help="overlap percentage between each cropped patch"
    )

    parser.add_argument(
        "--visible",
        type=bool,
        required=False,
        default=True,
        help="if visulaize the crop images and gts"
    )
    return parser.parse_args(args)


def main(args=None):
    """Main function for offline crop."""

    args = parse_args(args)
    datasetDir = args.dataset_path
    ori_img_dir = f'{datasetDir}/img/'
    assert os.path.isdir(ori_img_dir), f'Cannot find images dir: {ori_img_dir}'


    patch_h = args.patch_height
    patch_w = args.patch_width
    overlapRate =  args.overlapRate
    overlap_w = int(overlapRate * patch_w)
    overlap_h = int(overlapRate * patch_h)

    has_gt = args.has_gt
    if has_gt:
        assert os.path.isdir(f'{datasetDir}/gt/'), f"Cannot find gt dir: {f'{datasetDir}/gt/'}"

    visible = args.visible
    oriVis = f'{datasetDir}/vis'
    img_ext = args.img_ext

    patchImgDir = f'{datasetDir}/patch/img'
    patchGtDir = f'{datasetDir}/patch/gt'
    patchVisDir = f'{datasetDir}/patch/vis'
    os.makedirs(oriVis, exist_ok=True)
    os.makedirs(patchImgDir, exist_ok=True)
    os.makedirs(patchGtDir, exist_ok=True)
    os.makedirs(patchVisDir, exist_ok=True)


    for imgfile in glob.glob(f'{ori_img_dir}/*.{img_ext}'):

        oriImg = cv2.imread(imgfile)

        ori_h = oriImg.shape[0]
        ori_w = oriImg.shape[1]
        ori_c = oriImg.shape[2]

        # need to resize the ori image so that the there is no need to padding when crop the patch
        croppable_ori_w = int(math.ceil((ori_w - patch_w)/(patch_w - overlap_w)) *  (patch_w - overlap_w) + patch_w)
        croppable_ori_h = int(math.ceil((ori_h - patch_h)/(patch_h -overlap_h)) *  (patch_h -overlap_h) + patch_h)
        croppable_img, scale = keep_aspect_ratio_resize(oriImg, ori_w, ori_h, croppable_ori_w, croppable_ori_h)

        #  label 
        imgname = os.path.splitext( os.path.basename(imgfile))[0]
        imgext = os.path.splitext( os.path.basename(imgfile))[-1]
        if has_gt:
            labelfile = f'{datasetDir}/gt/gt_{imgname}.txt'
            annData = get_annotation(labelfile)
            text_polys = annData['text_polys']
            texts = annData['texts']


            text_polys[:, :, 0] *= scale
            text_polys[:, :, 1] *= scale
            text_polys = text_polys.astype(np.int32)
        # vis ori gt
        if visible:
            croppable_img_vis = copy.deepcopy(croppable_img)
            for bbox in text_polys:
                cv2.polylines(croppable_img_vis, [bbox], True, (0, 255, 0), 2) 
            cv2.imwrite(f'{oriVis}/{imgname}_vis.jpg', croppable_img_vis)

        num_col_cut = int((croppable_ori_w- patch_w)/(patch_w - overlap_w))
        num_raw_cut = int((croppable_ori_h- patch_h)/(patch_h - overlap_h))

        patch_gts = []
        for i in range(0, num_raw_cut+1):
            for j in range(0, num_col_cut+1):
                patch_gts.clear()

                x_start = int(j*(patch_w - overlap_w))
                y_start = int(i*(patch_h - overlap_h))
                patch = croppable_img[y_start : y_start + patch_h, x_start : x_start + patch_w, :]
                
                patchImgName = f'{patchImgDir}/{imgname}_{i}_{j}{imgext}'
                if os.path.isfile(patchImgName):
                    print(f'{patchImgName} alread exist, skip and continue')
                    continue
                cv2.imwrite(patchImgName,patch )

                if has_gt:
                    crop_x1 = x_start
                    crop_y1 = y_start
                    crop_x2 = x_start + patch_w
                    crop_y2 = y_start + patch_h
                    for gtIdx, gt in enumerate(text_polys):
                        pG = Polygon(gt)
                        pD = Polygon(((crop_x1,crop_y1),(crop_x2,crop_y1),(crop_x2,crop_y2),(crop_x1,crop_y2)))
                        intersection = pD.intersection(pG)
                        if intersection.is_empty:
                            continue
                        else:
                            patch_cur_gt = mapping(intersection)['coordinates'][0]
                            try:
                                if len(patch_cur_gt) == 5:
                                    patch_gts.append({'text':texts[gtIdx], 'poly':patch_cur_gt[:-1]})
                            except:
                                continue

                # save patch 
                if has_gt and len(patch_gts) > 0:
                    patch_polygons = np.asarray([gt['poly'] for gt in patch_gts], dtype=int)

                    patch_polygons[:,:,0] -= crop_x1
                    patch_polygons[:,:,1] -= crop_y1

                    if visible:
                        patch_copy = patch.copy()
                        for idx, bbox in enumerate(patch_polygons):
                            cv2.polylines(patch_copy, [bbox], True, (0, 255, 0), 2) 
                            cv2.putText(patch_copy, f"{patch_gts[idx]['text']}", tuple(np.min(bbox,axis=0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) , 2)
                        cv2.imwrite(f'{patchVisDir}/{imgname}_{i}_{j}_vis.jpg', patch_copy)

                patchGtName = f'{patchGtDir}/gt_{imgname}_{i}_{j}.txt'
                with open(patchGtName, 'w') as f:
                    for poly in patch_gts:
                        poly_points = [f'{int(points[0]-crop_x1)},{int(points[1]-crop_y1)}' for points in poly['poly']]
                        f.write(','.join(poly_points))
                        f.write(f",{poly['text']}\n")
    print(f'Offline crop done! results save to {datasetDir}/patch')

if __name__ == "__main__":    
    main()
