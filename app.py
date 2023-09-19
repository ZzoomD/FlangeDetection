import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import onnxruntime
from utils import *
from utils.general import *
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.augmentations import letterbox
import re

st.set_page_config(layout="wide")

cfg_model_path = './weights/23_9_11_pretrain_img1280.onnx'
model = None
confidence = .25
size = 640


def off_input(input_option="非实时图像/视频"):
    updata_bytes = st.sidebar.file_uploader("请上传非实时图像/视频", type=['png', 'jpeg', 'jpg', 'avi', 'mp4', 'mpv'])
    st.sidebar.write(f"已上传文件的路径: {updata_bytes.name}")
    if updata_bytes.name.split('.')[-1] in ['png', 'jpeg', 'jpg']:
        img_file = './data/upload_data/upload.' + updata_bytes.name.split('.')[-1]
        Image.open(updata_bytes).save(img_file)

        if img_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_file, caption="原始图像")
            with col2:
                img = process_frame(img_file, model, size, rename=0)
                st.image(img, caption="检测后图像")

    elif updata_bytes.name.split('.')[-1] in ['avi', 'mp4', 'mpv']:
        vid_file = './data/upload_data/upload.' + updata_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(updata_bytes.read())
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            output_img = process_frame(frame, model, size, rename=0)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()
    else:
        st.sidebar.write(f"不支持{updata_bytes.name}的格式")


def online_input(input_option="实时图像/视频"):
    num_cameras = get_camera_num()
    cameras_src = st.sidebar.multiselect("选择摄像头(可多选)", [str(i) + "号摄像头" for i in range(num_cameras)])
    cap_set = []
    for opt in cameras_src:
        cap_num = int(opt.split("号")[0])
        cap = cv2.VideoCapture(cap_num)
        cap.open(cap_num)
        cap_set.append([cap_num, cap])

    data_format = st.sidebar.radio("摄像头准备就绪，请选择待检测数据类型", ['图像', '视频'])

    if data_format == '图像':
        frame_set = []
        for cap in cap_set:
            # cap[1].open(cap[0])
            ret, frame = cap[1].read()
            if not ret:
                st.write(f"摄像头 {cap[0]} 无法读取帧")
                break
            frame_set.append(frame)
            cap[1].release()
        col = st.columns(len(frame_set))
        for i, frame in enumerate(frame_set):
            with col[i]:
                img = process_frame(frame, model, size, rename=0)
                st.image(img, caption=f"检测结果{i}")
    else:
        fps = 0
        fps_container = st.empty()
        fps_container.markdown(f"<span style='font-size: 24px;'>**FPS: {fps:.2f}**</span>", unsafe_allow_html=True)
        st.markdown("---")
        outputs = [st.empty() for _ in range(num_cameras)]
        while True:
            frame_set = []
            for cap in cap_set:
                ret, frame = cap[1].read()
                if not ret:
                    st.write(f"摄像头 {cap[0]} 无法读取帧")
                    break
                frame_set.append(frame)
            for i, frame in enumerate(frame_set):
                prev_time = time.time()
                img = process_frame(frame, model, size, rename=0)
                outputs[i].image(img, caption=f"检测结果{i}")
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                fps_container.markdown(f"<span style='font-size: 24px;'>**FPS: {fps:.2f}**</span>", unsafe_allow_html=True)

            key_pressed = cv2.waitKey(1)  # 每隔多少毫秒毫秒，获取键盘哪个键被按下
            if key_pressed in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
                break

        for cap in cap_set:
            cap[1].release()


def get_camera_num():
    num_cameras = 0
    while True:
        cap = cv2.VideoCapture(num_cameras)
        if not cap.isOpened():
            break
        num_cameras += 1
        cap.release()
    return num_cameras


def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(onnx_path, device=None):
    model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return model


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


# def get_user_model():
#     model_src = st.sidebar.radio("Model source", ["file upload", "url"])
#     model_file = None
#     if model_src == "file upload":
#         model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt', 'onnx'])
#         if model_bytes:
#             model_file = "weights/" + model_bytes.name
#             # with open(model_file, 'wb') as out:
#             #     out.write(model_bytes.read())
#     else:
#         url = st.sidebar.text_input("model url")
#         if url:
#             model_file_ = download_model(url)
#             if model_file_.split(".")[-1] == "pt":
#                 model_file = model_file_
#
#     return model_file

def get_user_model():
    model_bytes = st.sidebar.file_uploader("请上传自定义模型，文件命名格式为“640/1280_模型”", type=['pt', 'onnx'])
    if model_bytes:
        model_file = "./weights/" + model_bytes.name
    return model_file


def process_frame(img_bgr, ort_session, image_size=640, rename=None):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    palette = [
        ['background', [127, 127, 127]],
        ['green', [0, 255, 0]],
        ['red', [0, 0, 255]],
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

    if isinstance(img_bgr, str):
        img_bgr = cv2.imread(img_bgr)
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

    # ONNX Runtime预测
    # ONNX Runtime 输入
    ort_inputs = {'images': im}
    # onnx runtime 输出
    ort_output = ort_session.run(['output0'], ort_inputs)[0]
    ort_output = torch.from_numpy(ort_output)

    pred = non_max_suppression(ort_output, conf_thres=confidence, iou_thres=iou, max_det=1000)

    for i, det in enumerate(pred):  # per image

        # for save_crop
        annotator = Annotator(img_bgr, line_width=1, font_size=10, example=enable_classes)
        # annotator = Annotator(img_bgr, line_width=3, font_size=25, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_bgr.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if c not in enable_classes:
                    continue
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=palette_dict[c + 1])

        # Stream results
        img_bgr = annotator.result()

    # # 记录该帧处理完毕的时间
    # end_time = time.time()
    # # # 计算每秒处理图像帧数FPS
    # FPS = 1/(end_time - start_time)
    # #
    # # # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    # scaler = 2 # 文字大小
    # FPS_string = 'FPS {:.2f}'.format(FPS) # 写在画面上的字符串
    # img_bgr = cv2.putText(img_bgr, FPS_string, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

    return img_bgr

def main():
    # global variables
    global model, confidence, iou, cfg_model_path, size, enable_classes, names

    st.title("法兰孔衬套检测")

    st.sidebar.title("模型设置")

    # upload model
    model_src = st.sidebar.radio("检测模型", ["默认模型", "自定义模型"])
    # URL, upload file (max 200 mb)
    if model_src == "默认模型":
        model_src = st.sidebar.radio("模型尺寸", ["640", "1280"])
        if model_src == "640":
            user_model_path = os.path.join("./weights", model_src + "_模型" + ".onnx")
        else:
            user_model_path = os.path.join("./weights", model_src + "_模型"  + ".onnx")
    else:
        user_model_path = get_user_model()

    if user_model_path:
        cfg_model_path = user_model_path
        size = int(re.split(r"[/_\\]", cfg_model_path)[-2])

    # st.sidebar.text(cfg_model_path.split("/")[-1])
    st.sidebar.text(os.path.basename(cfg_model_path))
    st.sidebar.markdown("---")

    st.sidebar.title("参数设置")
    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("要求模型路径为“onnx”文件, 请重新选择模型文件.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("运行设备", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("运行设备", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        meta = model.get_modelmeta().custom_metadata_map  # metadata
        if 'stride' in meta:
            stride, names = int(meta['stride']), eval(meta['names'])
        names[0] = "正确(有衬套)"
        names[1] = "错误(无衬套)"
        names = list(names.values())

        # confidence slider
        confidence = st.sidebar.slider('置信度', min_value=0.1, max_value=1.0, value=.45)
        iou = st.sidebar.slider('交并比', min_value=0.1, max_value=1.0, value=.25)

        # custom classes
        if st.sidebar.checkbox("展示类别"):
            assigned_class = st.sidebar.multiselect("请选择需要展示的类别：", names, default=[names[1]])
            enable_classes = [names.index(name) for name in assigned_class]
        else:
            enable_classes = [i for i in range(len(names))]

        st.sidebar.markdown("---")

        st.sidebar.title("数据设置")
        # input options
        input_option = st.sidebar.radio("请选择待检测数据", ['非实时图像/视频', '实时图像/视频'])

        if input_option == '非实时图像/视频':
            off_input(input_option)
        else:
            online_input(input_option)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass