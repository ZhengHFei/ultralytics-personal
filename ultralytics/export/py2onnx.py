from ultralytics import YOLO
import onnxruntime
import cv2
import numpy as np
import torch
import time
import torchvision

import onnx_graphsurgeon as gs
import numpy as np
import onnx

conf_thres = 0.25
iou_thres = 0.7

def export_onnx():
    model = YOLO(r"D:\project\ultralytics\ultralytics\log\rail_height\train_rail_height\weights\last.pt")  # load a pretrained YOLOv8n model
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

def modify_onnx():
    onnx_path = "weights/seg/yolov8n-seg.onnx"
    save_path = "export/onnx/yolov8n_seg_modify.onnx"
    graph = gs.import_onnx(onnx.load(onnx_path))
        
    target_nodes_name = ["/model.22/cv4.0/cv4.0.2/Conv", "/model.22/cv4.1/cv4.1.2/Conv", "/model.22/cv4.2/cv4.2.2/Conv", \
                    "/model.22/Concat_1", "/model.22/Concat_2", "/model.22/Concat_3"]
    target_nodes_index = []
    target_nodes = []
    for i, node in enumerate(graph.nodes):
        # print(node.name)
        if node.name in target_nodes_name:
            print(node.name)
            target_nodes_index.append(i)
            target_nodes.append(node)
    print(target_nodes_index)
    for i in target_nodes_index:
        dependent_nodes = [node for node in graph.nodes[i + 1:] if  graph.nodes[i].outputs[0] in node.inputs]
    
    # print(dependent_nodes)
    # for n in dependent_nodes:
    #     graph.nodes.remove(n)
    
    # output
    proto = graph.outputs[1]
    proto.name = "proto"
        
    mask_8 = gs.Variable(name="ouput1", dtype=target_nodes[0].outputs[0].dtype, shape=target_nodes[0].outputs[0].shape)
    mask_16 = gs.Variable(name="mask_16", dtype=target_nodes[1].outputs[0].dtype, shape=target_nodes[1].outputs[0].shape)
    mask_32 = gs.Variable(name="mask_32", dtype=target_nodes[2].outputs[0].dtype, shape=target_nodes[2].outputs[0].shape)
    detect_8 = gs.Variable(name="detect_8", dtype=target_nodes[3].outputs[0].dtype, shape=target_nodes[3].outputs[0].shape)
    detect_16 = gs.Variable(name="detect_16", dtype=target_nodes[4].outputs[0].dtype, shape=target_nodes[4].outputs[0].shape)
    detect_32 = gs.Variable(name="detect_32", dtype=target_nodes[5].outputs[0].dtype, shape=target_nodes[5].outputs[0].shape)
    
    target_nodes[0].outputs = [mask_8]
    target_nodes[1].outputs = [mask_16]
    target_nodes[2].outputs = [mask_32]
    target_nodes[3].outputs = [detect_8]
    target_nodes[4].outputs = [detect_16]
    target_nodes[5].outputs = [detect_32]
    
    graph.outputs = [ proto, mask_8, mask_16, mask_32, detect_8, detect_16, detect_32 ]

    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph),save_path)

def modify_abrasion_onnx():
    onnx_path = "../log/rail_height/train_rail_height/weights/last.onnx"
    save_path = "onnx/rail_height_ax650.onnx"
    graph = gs.import_onnx(onnx.load(onnx_path))
        
    target_nodes_name = ["/model.22/dfl/conv/Conv", "/model.22/Sigmoid"]
    target_nodes_index = []
    target_nodes = []
    for i, node in enumerate(graph.nodes):
        # print(node.name)
        if node.name in target_nodes_name:
            print(node.name)
            target_nodes_index.append(i)
            target_nodes.append(node)
    print(target_nodes_index)
    
    bbox = gs.Variable(name="bbox", dtype=target_nodes[0].outputs[0].dtype, shape=target_nodes[0].outputs[0].shape) 
    prob = gs.Variable(name="prob", dtype=target_nodes[1].outputs[0].dtype, shape=target_nodes[1].outputs[0].shape)
       
    target_nodes[0].outputs = [bbox]
    target_nodes[1].outputs = [prob]
    graph.outputs = [ prob, bbox]
    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph),save_path)

if __name__ == "__main__":
    
    # export_onnx()
    # modify_onnx()
    modify_abrasion_onnx()
