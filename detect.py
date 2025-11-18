# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
arduino_mode = 1 # default is 1 -> follow ball
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
from ultralytics.utils.plotting import Annotator, colors, save_one_box


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

import serial
import time

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data_train/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data_train/data_train.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

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
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    #khởi tạo cổng giao tiếp
    try:
        arduino = serial.Serial('COM7', 115200, timeout=1)
        time.sleep(2)
        print("Kết nối Arduino thành công.")
        #mode = 1  # mode mặc định là 1
    except Exception as e:
        arduino = None
        #mode = 1
        print(f"Không thể mở cổng serial: {e}")

    #kết thúc khởi tạo

    # ==================== check mode from mega->nano-> pc ====================


    def update_arduino_mode():
        global arduino_mode
        if arduino and arduino.in_waiting:
            try:
                line = arduino.readline().decode(encoding='utf-8', errors='ignore').strip()
                print(f"[Serial] Nhận được: {line}")  # debug
                if line.startswith("MODE:"):
                    mode_val = int(line.split(":")[1])
                    if mode_val in [1, 2]:
                        arduino_mode = mode_val  # ✅ đúng rồi!
                        print(f"[Serial] Cập nhật Arduino MODE = {arduino_mode}")
            except Exception as e:
                print(f"Lỗi đọc chế độ từ Arduino: {e}")

    #update mode
    for path, im, im0s, vid_cap, s in dataset:
        update_arduino_mode()  # đọc chế độ từ nano
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                saw_ball = False  # có thấy class bong_tennis (class 0) trong frame này không?
                # (nếu cần, có thể thêm saw_khu = False cho khu_chua_bong)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # #lọc
                # # Sắp xếp theo độ tin cậy (confidence)
                # det = det[det[:, 4].argsort(descending=True)]
                # # Chỉ lấy 1 đối tượng đầu tiên
                # det = det[:1]
                # # lọc đến đâyyy

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # nếu tháy bong s thì đánh dấu
                    if c == 0:
                        saw_ball = True

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # === VẼ HỘP + CHIA MÀN HÌNH ===
                    if save_img or save_crop or view_img:
                        # Trung tâm vật thể
                        x1, y1, x2, y2 = xyxy
                        center_x = (x1 + x2) / 2
                        # Kích thước ảnh
                        h, w = im0.shape[:2]
                        region_width = w / 9

                        # Xác định vùng từ -4 -> 4
                        region_index = int((center_x // region_width) - 4)
                        region_index = max(-4, min(4, region_index))
                        # gửi tọa độ
                        # GỬI TỌA ĐỘ XUỐNG ARDUINO THEO DẠNG: "<class_id> <region_index>\n"
                        if arduino:
                            try:
                                should_send = False
                                if c == 1:
                                    print("")
                                    #Không gửi gì nếu là con_nguoi
                                    #print(f"[Skip] Bỏ qua class con_nguoi ({c})")
                                elif arduino_mode == 1 and c == 0:
                                    should_send = True
                                elif arduino_mode == 2 and c == 2:
                                    should_send = True

                                if should_send:
                                    data = f"{c} {region_index}\n"
                                    arduino.write(data.encode())
                                    print(f"[Send] MODE {arduino_mode} - Gửi: {data.strip()}")
                                else:
                                    #print(f"[Skip] MODE {arduino_mode} - Bỏ qua class {c}")
                                     print("")
                            except Exception as e:
                                print(f"Lỗi gửi Serial: {e}")
                        #kết thúc đoạn guiwr
                        # Ghi nhãn có vị trí vùng
                        label = None if hide_labels else f"{names[c]} {conf:.2f} pos:{region_index}"

                        # Vẽ bounding box
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # === VẼ ĐƯỜNG CHIA MÀN HÌNH ===
                    h, w = im0.shape[:2]
                    step = w // 9
                    for i in range(1, 9):
                        x = int(step * i)
                        cv2.line(im0, (x, 0), (x, h), (0, 255, 255), 1)  # Đường chia màu vàng

                    # Ghi số vùng (-4 đến 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for i in range(9):
                        text = str(i - 4)
                        x = int(step * (i + 0.5))
                        cv2.putText(im0, text, (x - 10, 30), font, 0.7, (255, 255, 0), 2)

            # Sau khi xử lý xong tất cả bbox trong frame này:
            # không thấy bsng thì quay phải setpoint = 3
            # Fallback chỉ hoạt động khi đang ở chế độ theo dõi bóng
            if arduino and arduino_mode == 1:
                try:
                    if 'saw_ball' not in locals() or not saw_ball:
                        fallback_region = 2  # quay phải
                        data = f"0 {fallback_region}\n"  # class 0 (bong_tennis), region = 2
                        arduino.write(data.encode())
                        #print(f"[Fallback] MODE 1 - Không thấy bóng, gửi lệnh quay phải: {data.strip()}")
                except Exception as e:
                    print(f"Lỗi gửi Serial fallback: {e}")
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    # Vẽ chia màn hình thành 9 phần (-4 đến 4, 0 là center)
                    h, w, _ = im0.shape
                    third_w = w // 3
                    third_h = h // 3

                    # Vẽ 2 đường dọc chia 3 phần
                    for i in range(1, 3):
                        x = i * third_w
                        cv2.line(im0, (x, 0), (x, h), (0, 255, 0), 1)

                    # Vẽ 2 đường ngang chia 3 phần
                    for j in range(1, 3):
                        y = j * third_h
                        cv2.line(im0, (0, y), (w, y), (0, 255, 0), 1)

                    # Vẽ trục tọa độ -4 đến 4 (giả định)
                    center_x, center_y = w // 2, h // 2
                    cv2.line(im0, (center_x, 0), (center_x, h), (0, 0, 255), 1)  # Trục dọc giữa
                    cv2.line(im0, (0, center_y), (w, center_y), (0, 0, 255), 1)  # Trục ngang giữa

                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'DD
                    # Đảm bảo danh sách đủ phần tử để tránh lỗi IndexError
                    while len(vid_path) <= i:
                        vid_path.append('')
                    while len(vid_writer) <= i:
                        vid_writer.append(None)

                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
