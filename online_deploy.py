import time
import numpy as np
import cv2

import onnxruntime
from utils import *
from utils.general import *
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.augmentations import letterbox

import matplotlib.pyplot as plt
#matplotlib inline


def process_frame(img_bgr, ort_session, image_size=640, rename=None):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    palette = [
        ['background', [127, 127, 127]],
        ['green', [0, 200, 0]],
        ['red', [0, 0, 200]],
        ['white', [144, 238, 144]],
        ['seed-black', [30, 30, 30]],
        ['seed-white', [8, 189, 251]]
    ]

    palette_dict = {}
    for idx, each in enumerate(palette):
        palette_dict[idx] = each[1]

    # 记录该帧开始处理的时间
    start_time = time.time()

    meta = ort_session.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])

    if rename is not None:
        names[0] = "Correct"
        names[1] = "Error"

    im = letterbox(img_bgr, image_size, stride=stride, auto=False)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im)
    im = im.half() if not ort_session else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]
    if not ort_session and im.dtype != torch.float16:
        im = im.half()
    im = im.numpy()

    # # 缩放至模型要求的像素
    # img_bgr_resize = cv2.resize(img_bgr, (image_size, image_size))  # 缩放尺寸
    #
    # # 预处理
    # img_tensor = img_bgr_resize
    # mean = (123.675, 116.28, 103.53)  # BGR 三通道的均值
    # std = (58.395, 57.12, 57.375)  # BGR 三通道的标准差
    #
    # # 归一化
    # img_tensor = (img_tensor - mean) / std
    # img_tensor = img_tensor.astype('float32')
    # img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    # img_tensor = np.transpose(img_tensor, (2, 0, 1))  # 调整维度
    # input_tensor = np.expand_dims(img_tensor, axis=0)  # 扩充 batch-size 维度

    # ONNX Runtime预测
    # ONNX Runtime 输入
    ort_inputs = {'images': im}
    # onnx runtime 输出
    ort_output = ort_session.run(['output0'], ort_inputs)[0]
    ort_output = torch.from_numpy(ort_output)

    pred = non_max_suppression(ort_output, conf_thres=0.20, iou_thres=0.35, max_det=1000)

    for i, det in enumerate(pred):  # per image

        # for save_crop
        annotator = Annotator(img_bgr, line_width=1, font_size=10, example=str(names))
        # annotator = Annotator(img_bgr, line_width=3, font_size=25, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_bgr.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=palette_dict[c + 1])
                # if c == 0:
                #     annotator.box_label(xyxy, label, color=colors(16, True))
                #
                # else:
                #     annotator.box_label(xyxy, label, color=colors(18, True))

        # Stream results
        img_bgr = annotator.result()

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)
    #
    # # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    scaler = 2 # 文字大小
    FPS_string = 'FPS {:.2f}'.format(FPS) # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

    return img_bgr


if __name__ == '__main__':
    # 亮度调整
    bright_threshold = 20
    brightness_increase = 15
    image_size = 640
    # ONNX 模型路径
    onnx_path = './weights/23_9_14_pretrain_img640.onnx'
    # onnx_path = './weights/best.onnx'
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # online camera
    # cap = cv2.VideoCapture(1)
    # cap.open(0)

    # # offline video
    # cap = cv2.VideoCapture('./dataset/demo1.mp4')
    cap = cv2.VideoCapture('./dataset/demo2_2.avi')

    # # offline image
    # cap = cv2.VideoCapture('./dataset/flange_w_wo_bush_side(20).jpeg')

    # show window
    cv2.namedWindow('Results_Window', cv2.WINDOW_NORMAL)

    # 无限循环，直到break被触发
    while cap.isOpened():

        # 获取画面
        success, frame = cap.read()

        if not success:  # 如果获取画面不成功，则退出
            print('获取画面不成功，退出')
            break

        # 预处理
        # 计算图像平均亮度
        average_bright = np.mean(frame)
        if average_bright < bright_threshold:
            # 分割通道
            b, g, r = cv2.split(frame)

            # 增加每个通道的亮度
            b = np.clip(b + brightness_increase, 0, np.max(b)).astype(np.uint8)
            g = np.clip(g + brightness_increase, 0, np.max(g)).astype(np.uint8)
            r = np.clip(r + brightness_increase, 0, np.max(r)).astype(np.uint8)

            # 合并通道
            frame = cv2.merge((b, g, r))

        ## 逐帧处理
        frame = process_frame(frame, ort_session, image_size, rename=0)

        # 展示处理后的三通道图像
        cv2.imshow('Results_Window', frame)

        key_pressed = cv2.waitKey(1)  # 每隔多少毫秒毫秒，获取键盘哪个键被按下
        # print('键盘上被按下的键：', key_pressed)

        if key_pressed in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break

    # 关闭摄像头
    cap.release()

    # 关闭图像窗口
    cv2.destroyAllWindows()