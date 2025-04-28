from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os, sys
import argparse
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['QT_QPA_PLATFORM']="offscreen"
import torch
from groundingdino.util import box_ops
# from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
# import groundingdino.datasets.transforms as T
# from typing import Tuple, List

import numpy as np
# import tifffile as tiff

sys.path.append("./")

# segment anything
sys.path.append("..")
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from data.label import get_labels
from data.scene_config import get_scene_config
import json


def load_models(args):
    """
    Load DINO and SAM models
    
    Args:
        args: Command line arguments object
    
    Returns:
        dino_model, sam_predictor: Loaded models
    """
    dino_model = load_model(args.dino_config, args.dino_model)
    sam = build_sam(checkpoint=args.sam_checkpoint)
    sam = sam.cuda()
    sam_predictor = SamPredictor(sam)
    return dino_model, sam_predictor

def generate_mask(img_name, args, dino_model, sam_predictor, config, id2label, Thing_sem, Segment_sem):
    """
    Generate masks for an image
    
    Args:
        img_name: Image name
        args: Command line arguments object
        dino_model: DINO model
        sam_predictor: SAM predictor
        config: Scene configuration
        id2label: Label mapping
        Thing_sem: Thing class semantic information
        Segment_sem: Segmentation semantic information
    """
    path = os.path.join(config["image_path"], img_name)
    image_source, image = load_image(path)

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=config["text_prompt"],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        remove_combined=True
    )

    idx = [i for i in range(len(phrases)) if phrases[i] == ''] 
    bbox_mask = torch.ones(boxes.shape[0]).bool()  
    bbox_mask[idx] = 0
    boxes = boxes[bbox_mask[:,None].expand(-1,boxes.shape[1])].reshape(-1,4)
    logits[bbox_mask]
    phrases = [phrases[i] for i in range(len(phrases)) if phrases[i] != '']

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # cv2.imwrite("groundSAM/bbox/annotated_image_.jpg", annotated_frame)
    sam_predictor.set_image(image_source)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
    # print(transformed_boxes.shape)

    semantic_label, logits_map, instance_label, instance_label_good = np.zeros([H,W]), np.zeros([H,W]), np.zeros([H,W]), np.zeros([H,W])
    instance_id, instance_id_good = 1, 1 # 1 is the stuff
    for i in range(transformed_boxes.shape[0]):
        # try:
        if (transformed_boxes[i][2].item()-transformed_boxes[i][0].item())>W*0.7 and (transformed_boxes[i][3].item()-transformed_boxes[i][1].item())>H*0.7:
            continue
        else:
            masks, de1, de2 = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes[i][None,:],
                        multimask_output = False,
                    )
            # except:
            #     return None, None, None
            if phrases[i] in config["class_name"]:
                mask_now = masks[0][0].cpu().numpy()
                c_mask = logits[i].item()*mask_now > logits_map
                class_label = config["class_label"][config["class_name"].index(phrases[i])] 
                semantic_label[c_mask] = class_label
                logits_map[c_mask] = logits[i]
                # THING or STUFF
                if config["thing"][class_label-1]:
                    instance_label[c_mask] = instance_id
                    instance_id += 1
                    if logits[i]>args.instance_threshold: 
                        instance_label_good[c_mask] = instance_id_good
                        instance_id_good += 1
                        Thing_sem[img_name].append(class_label)
                # elif ~THING[class_label-1]:
                #     instance_label[c_mask] = 1 # stuff

    ## semantic     
    frame_with_semantic = show_label(semantic_label.astype(np.int8), annotated_frame, id2label) 
    folder_semanitc = config["save_path"] + "/semantic_image"
    os.makedirs(folder_semanitc, exist_ok=True)
    os.makedirs(folder_semanitc+'_rgb', exist_ok=True)

    ## instance     
    frame_with_instance = show_label(instance_label.astype(np.int8), annotated_frame, id2label, mode='instance') 
    folder_instance = config["save_path"] + "/instance_image"
    os.makedirs(folder_instance, exist_ok=True)
    os.makedirs(folder_instance+'_rgb', exist_ok=True)

    cv2.imwrite(os.path.join(folder_semanitc+'_rgb', f"{img_name}"), frame_with_semantic)
    cv2.imwrite(os.path.join(folder_semanitc, f"{img_name}"), semantic_label.astype(np.int8))
    
    cv2.imwrite(os.path.join(folder_instance+'_rgb', f"{img_name}"), frame_with_instance)
    cv2.imwrite(os.path.join(folder_instance, f"{img_name}"), instance_label.astype(np.int8))


def show_label(mask, image, id2label, mode='semantic'):
    if mode == 'semantic':
        v_colors = np.vstack([id2label[semID].color for semID in mask.reshape(-1).tolist()])
        h, w = mask.shape[-2:]
        mask_image = v_colors.reshape(h, w, 3)
        return cv2.addWeighted(image,0.4,mask_image.astype(np.uint8),0.6,0)
    
    elif mode == 'instance':
        v_colors = np.vstack([id2rgb(ID) for ID in mask.reshape(-1).tolist()])
        h, w = mask.shape[-2:]
        mask_image = v_colors.reshape(h, w, 3)
        return mask_image


def id2rgb(id):
    # Convert ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3,), dtype=np.uint8)
    if id==0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)
    return rgb


def parse_args():
    parser = argparse.ArgumentParser(description='Ground SAM mask generation script')
    parser.add_argument('--dataset', type=str, default='scannet', help='Dataset name (scannet, scannet++, etc.)')
    parser.add_argument('--scene', type=str, default='0087_02', help='Scene ID')
    parser.add_argument('--dataset_root', type=str, default='/mnt/nas_new/yx/dataset/', help='Dataset root path')
    parser.add_argument('--box_threshold', type=float, default=0.2, help='Box threshold for GroundingDINO')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold for GroundingDINO')
    parser.add_argument('--instance_threshold', type=float, default=0.35, help='Instance threshold')
    parser.add_argument('--dino_config', type=str, default="third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help='GroundingDINO config file path')
    parser.add_argument('--dino_model', type=str, default="third-party/GroundingDINO/groundingdino_swint_ogc.pth", help='GroundingDINO model weights path')
    parser.add_argument('--sam_checkpoint', type=str, default='third-party/segment_anything/sam_vit_h_4b8939.pth', help='SAM model checkpoint path')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load models
    dino_model, sam_predictor = load_models(args)
    
    # Get scene configuration
    config = get_scene_config(args.dataset, args.scene, args.dataset_root)
    
    # Get labels
    id2label = get_labels(args.scene)
    
    # Process images
    images = os.listdir(config["image_path"])
    images = sorted(images, key=lambda x: int(x.split('.')[0]))
    Thing_sem = {}  # For instance images
    Segment_sem = {}
    
    for img_name in tqdm(images):
        Thing_sem[img_name] = []
        generate_mask(
            img_name, 
            args, 
            dino_model, 
            sam_predictor, 
            config, 
            id2label, 
            Thing_sem, 
            Segment_sem
        )
    
    # Save results
    with open(os.path.join(config["save_path"], "thing_semantic_label.json"), "w") as f:
        json.dump(Thing_sem, f)
    with open(os.path.join(config["save_path"], "segment_semantic_label.json"), "w") as f:
        json.dump(Segment_sem, f)

    