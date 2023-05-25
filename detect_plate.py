import argparse, os, platform, sys, time
from pathlib import Path
import torch
import cv2

#import modules from yolov5
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def detect(
        weights='models/yolov5s_custom.pt',  # model path or triton URL
        source='input',  # file/dir/URL/glob/screen/0(webcam)
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.75,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device=torch.device('cpu'),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project='output',  # save results to project/name
        line_thickness=3,  # bounding box thickness (pixels)
        vid_stride=1,  # video frame-rate stride
        nosave=False,  # do not save inference images
        save_crop=True,  # save cropped prediction boxes
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        save_txt=False,  # save results to *.txt
):
        # Initialize
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        webcam = source.isnumeric() or source.endswith('.streams')
        screenshot = source.lower().startswith('screen')

        # Directories
        save_dir = increment_path(Path(project), exist_ok=True)  # increment run
        (save_dir).mkdir(parents=False, exist_ok=True)  # make dir


        #load Model
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # # Initialize model
        # model = torch.load(f, path=weights, source='local')  # local repo

        # Dataloader
        bs = 1  # batch_size
        if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
        elif screenshot:
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim

                        # Inference
                        with dt[1]:
                                # visualize = increment_path(save_dir / Path(path).stem, mkdir=False) if visualize else False
                                pred = model(im, augment=False, visualize=False)

                        # NMS
                        with dt[2]:
                                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, max_det=1000)

                # Process predictions
                for i, det in enumerate(pred):  # detections per image
                        seen += 1
                        if webcam:  # batch_size >= 1
                                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                                s += f'{i}: '
                        else:
                                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))        
                        if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, 5].unique():
                                        n = (det[:, 5] == c).sum()  # detections per class
                                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                
                                # Remove the existing file if it exists
                                if os.path.exists(f'output/images.jpg'):
                                        os.remove(f'output/images.jpg')

                                # Write results to file and replace current one
                                for *xyxy, conf, cls in reversed(det):
                                        if save_txt :  # Write to file
                                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                                line = (cls, *xywh, conf) # label format
                                                with open(f'{txt_path}.txt', 'w') as f:
                                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                        if save_img or save_crop or view_img:  # Add bbox to image
                                                c = int(cls)  # integer class
                                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                                annotator.box_label(xyxy, label, color=colors(c, True))
                                        if save_crop:
                                                save_one_box(xyxy, imc, file=save_dir /  'images.jpg', BGR=False)

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

        
detect()