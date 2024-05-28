from ultralytics import YOLO
import onnxruntime
import cv2
import numpy as np
import torch
import time
import torchvision

conf_thres = 0.25
iou_thres = 0.7

def export_onnx():
    model = YOLO("log/scale_31_neg_aug/last.pt")  # load a pretrained YOLOv8n model
    model.export(format="onnx", half=False, simplify=True)  # export the model to ONNX forma

def preprocess(image, size):
    h, w, _ = image.shape
    r_h = size[0] / h
    r_w = size[1] / w
    r = min(r_h, r_w)
    
    image = cv2.resize(image, (int(w * r), int(h * r)))
    input_image = np.full((size[0], size[1], 3), [114, 114, 114], dtype=np.uint8)
    x = (size[1]-image.shape[1])//2
    y = (size[0]-image.shape[0])//2
    input_image[y:y+image.shape[0], x:x+image.shape[1]] = image
    # cv2.imshow("image", input_image)
    # cv2.waitKey(0)
    return input_image/255.

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def postprocess(prediction,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=300,
        max_wh=7680,
):

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # 0
    print('nm: ', nm)
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    print('xc: ', xc)

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [np.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = np.concatenate((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        # if classes is not None:
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

# def nms():
    

def test_onnx():
    onnx_path = "log/weight/best.onnx"
    # output 1x12x5376
    # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    provider = "CUDAExecutionProvider"
    onnx_session = onnxruntime.InferenceSession(onnx_path, providers=[provider])

    for ii in range(30):
        t1 = time.time()
        image = cv2.imread("test_images/images/03699_2.jpg")
        
        input_tensors = onnx_session.get_inputs()
        output_tensors = onnx_session.get_outputs() 
        
        shape = input_tensors[0].shape
        input_image = preprocess(image, shape[2:])
        input_image = input_image.transpose(2, 0, 1).astype(np.float32)[None, ...]
        outputs = onnx_session.run(None, {input_tensors[0].name: input_image})
        print(outputs[0].shape)
        t2 = time.time()
        print(f"Time: {(t2-t1)*1000}ms")
    
    
def test():
    model = YOLO("log/weight/best.pt")
    image_list = [
                # 'test_images/images/00001.jpg',\
                # 'test_images/images/00021_1.jpg', \
                # 'test_images/images/01276_1.jpg', \
                # 'test_images/images/03600_1.jpg', \
                'test_images/images/03699_2.jpg'\
                ]
    results = model(image_list)
    # 在帧上可视化结果
    for ii in range(len(image_list)):
        annotated_frame = results[ii].plot()
        boxes = results[ii].boxes  # 边界框输出的 Boxes 对象
        # print("=======boxs======")
        # print(boxes)
        cls = boxes.cls.cpu().numpy()
        cls += 1
        print(cls)
        xywh = boxes.xywh.cpu().numpy()
        d12 = None
        d23 = None
        d15 = None
        d56 = None
        d57 = None
        if 2 in cls:
            indices2 = np.where(cls == 2)[0][0]
            if 1 in cls:
                indices1 = np.where(cls == 1)[0][0]
                d12_x = abs(xywh[indices1][0] - xywh[indices2][0])
                d12_y = abs(xywh[indices1][1] - xywh[indices2][1])
                d12 = np.sqrt(d12_x ** 2 + d12_y ** 2)
            if 3 in cls:
                indices3 = np.where(cls == 3)[0][0]
                d23_x = abs(xywh[indices3][0] - xywh[indices2][0])
                d23_y = abs(xywh[indices3][1] - xywh[indices2][1])
                d23 = np.sqrt(d23_x ** 2 + d23_y ** 2)
        if 5 in cls:
            indices5 = np.where(cls == 5)[0][0]
            if 1 in cls:
                indices1 = np.where(cls == 1)[0][0]
                d15_x = abs(xywh[indices1][0] - xywh[indices5][0])
                d15_y = abs(xywh[indices1][1] - xywh[indices5][1])
                d15 = np.sqrt(d15_x ** 2 + d15_y ** 2)
            if 6 in cls:
                indices6 = np.where(cls == 6)[0][0]
                d56_x = abs(xywh[indices6][0] - xywh[indices5][0])
                d56_y = abs(xywh[indices6][1] - xywh[indices5][1])
                d56 = np.sqrt(d56_x ** 2 + d56_y ** 2)
            if 7 in cls:
                indices7 = np.where(cls == 7)[0][0]
                d57_x = abs(xywh[indices7][0] - xywh[indices5][0])
                d57_y = abs(xywh[indices7][1] - xywh[indices5][1])
                d57 = np.sqrt(d57_x ** 2 + d57_y ** 2)

        cv2.putText(annotated_frame, f"d12: {d12:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f"d23: {d23:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f"d15: {d15:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f"d56: {d56:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated_frame, f"d57: {d57:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 显示带注释的帧
        cv2.imshow("YOLOv8推理", annotated_frame)
        cv2.imwrite(f"test_images/result/{ii}.jpg", annotated_frame)
        cv2.waitKey(0)
       

def testVideo():
    model = YOLO("log/scale_31_neg_aug/last.pt")
    # 打开视频文件
    video_path = r"D:\project\ultralytics\ultralytics\test_images\images\834_BINO_20231030_164415_312_clip.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    
    # 设置输出视频文件的参数
    output_file = r'D:\project\ultralytics\ultralytics\test_images\result\834_BINO_20231030_164415_312_clip.mp4'
    frame_size = frame.shape[:2]  # 每帧的大小

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以尝试其他编码器
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size[::-1])
    while success:
        t1 = time.time()
        h = 256
        w = 512
        sx = 300
        sy = 350
        # sx = 440
        # sy = 330
        image = frame[sy:sy+h, sx:sx+w]
        results = model(image)
        annotated_frame = results[0].plot()
        
        boxes = results[0].boxes  # 边界框输出的 Boxes 对象
        cls = boxes.cls.cpu().numpy()
        cls += 1
        # print(cls)
        xywh = boxes.xywh.cpu().numpy()
        d12 = -1
        d23 = -1
        d15 = -1
        d56 = -1
        d57 = -1
        if 2 in cls:
            indices2 = np.where(cls == 2)[0][0]
            if 1 in cls:
                indices1 = np.where(cls == 1)[0][0]
                d12_x = abs(xywh[indices1][0] - xywh[indices2][0])
                d12_y = abs(xywh[indices1][1] - xywh[indices2][1])
                d12 = np.sqrt(d12_x ** 2 + d12_y ** 2)
            if 3 in cls:
                indices3 = np.where(cls == 3)[0][0]
                d23_x = abs(xywh[indices3][0] - xywh[indices2][0])
                d23_y = abs(xywh[indices3][1] - xywh[indices2][1])
                d23 = np.sqrt(d23_x ** 2 + d23_y ** 2)
        if 5 in cls:
            indices5 = np.where(cls == 5)[0][0]
            if 1 in cls:
                indices1 = np.where(cls == 1)[0][0]
                d15_x = abs(xywh[indices1][0] - xywh[indices5][0])
                d15_y = abs(xywh[indices1][1] - xywh[indices5][1])
                d15 = np.sqrt(d15_x ** 2 + d15_y ** 2)
            if 6 in cls:
                indices6 = np.where(cls == 6)[0][0]
                d56_x = abs(xywh[indices6][0] - xywh[indices5][0])
                d56_y = abs(xywh[indices6][1] - xywh[indices5][1])
                d56 = np.sqrt(d56_x ** 2 + d56_y ** 2)
            if 7 in cls:
                indices7 = np.where(cls == 7)[0][0]
                d57_x = abs(xywh[indices7][0] - xywh[indices5][0])
                d57_y = abs(xywh[indices7][1] - xywh[indices5][1])
                d57 = np.sqrt(d57_x ** 2 + d57_y ** 2)

        frame[sy:sy+h, sx:sx+w] = annotated_frame
        cv2.putText(frame, f"d12: {d12:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"d23: {d23:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"d15: {d15:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"d56: {d56:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"d57: {d57:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(frame, (sx, sy), (sx + w, sy + h), (0, 255, 0), 2)
        
        print(f"all time:{(time.time() - t1)*1000}ms")
        video_writer.write(frame)
        # cv2.imshow("YOLOv8推理", frame)
        # cv2.waitKey(0) 
        # exit()
        success, frame = cap.read() 
    
    video_writer.release()
    cap.release()


def test_seg():
    model = YOLO("weights/seg/yolov8n-seg.pt")  # load a pretrained YOLOv8n model
    # model = YOLO('log/segment/train/weights/last.pt')
    results = model(["test_images/images/245.jpg", 
                    "test_images/images/317.jpg", 
                    "test_images/images/image.jpg", 
                    "test_images/images/408.jpg", 
                    "test_images/images/493.jpg"], save=True, show=True)

def test_abrasion():
    model = YOLO('log/abrasion/train_aug/weights/last.pt')
    results = model([
                    "test_images/abrasion/test.jpg"
                    # "test_images/abrasion/00000_640.jpg", 
                    # "test_images/abrasion/00000.jpg", 
                    # "test_images/abrasion/00004.jpg", 
                    # "test_images/abrasion/00005.jpg", 
                    # "test_images/abrasion/00056.jpg", 
                    # "test_images/abrasion/00089.jpg", 
                    # "test_images/abrasion/00153.jpg", 
                    # "test_images/abrasion/00157.jpg", 
                    # "test_images/abrasion/00160.jpg", 
                    # "test_images/abrasion/00006.jpg", 
                    # "test_images/abrasion/00007.jpg", 
                    # "test_images/abrasion/00016.jpg", 
                    ], save=True, show=False)

def test_flexible():
    model = YOLO('log/abrasion/train_flexible_noM/weights/last.pt')
    results = model([
                    "test_images/flexible/0000.jpg",
                    "test_images/flexible/0048.jpg",
                    "test_images/flexible/0473.jpg",
                    "test_images/flexible/0618.jpg",
                    "test_images/flexible/0998.jpg"
                    ], save=True, show=False)

def test_rail_height():
    model = YOLO('log/rail_height/train_rail_height/weights/last.pt')
    results = model([
                    "test_images/rail_height/00.bmp",
                    "test_images/rail_height/sun.bmp",
                    # "test_images/rail_height/00220.jpg",
                    # "test_images/rail_height/00255.jpg",
                    # "test_images/rail_height/00470.jpg",
                    # "test_images/rail_height/00595.jpg"
                    ], save=True, show=False)

if __name__ == "__main__":
    # export_onnx()
    # test_onnx()
    # testVideo()
    # test_seg()
    # test_abrasion()
    # test_flexible()
    test_rail_height()
