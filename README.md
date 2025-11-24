#include "PID_v1.h"

// ================== CẢM BIẾN ==================
const int ballSensorPin = A15;
const int magnetSensor1Pin = 24;
const int magnetSensor2Pin = 26;

int irSensorState = LOW;
int lastIrReading = LOW;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 10;
int ballCount = 0;

// ================== CHẾ ĐỘ ==================
int mode = 1;
bool pcCommandReceived = false;
unsigned long lastCommandTime = 0;
const unsigned long commandTimeout = 2000;

// ================== NÉ NGƯỜI ==================
bool avoidingPerson = false;
unsigned long avoidStartTime = 0;
const unsigned long avoidDuration = 500;

// ================== BTS7960 ==================
const int LPWM_LEFT  = 12;
const int RPWM_LEFT  = 13;
const int LPWM_RIGHT = 10;
const int RPWM_RIGHT = 11;
const int LPWM_SHOOT = 8;
const int RPWM_SHOOT = 9;
const int EN_ALL     = 22;

// ================== TỐC ĐỘ ==================
const int baseSpeed = 60;
const int minSpeed = 30;
const int maxSpeed = 120;
const int avoidSpeed = 100;
const int speedDiffPerRegion = 15;

double setpoint = 5;
unsigned long lastModeReport = 0;

// ================== MOTOR BẮN (SOFT) ==================
int shootPWM = 0;
int shootPWM_Target = 0;
unsigned long lastShootUpdate = 0;

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);
  Serial3.begin(9600);

  pinMode(LPWM_LEFT, OUTPUT);
  pinMode(RPWM_LEFT, OUTPUT);
  pinMode(LPWM_RIGHT, OUTPUT);
  pinMode(RPWM_RIGHT, OUTPUT);
  pinMode(RPWM_SHOOT, OUTPUT);
  pinMode(LPWM_SHOOT, OUTPUT);
  pinMode(EN_ALL, OUTPUT);

  pinMode(ballSensorPin, INPUT);
  pinMode(magnetSensor1Pin, INPUT_PULLUP);
  pinMode(magnetSensor2Pin, INPUT_PULLUP);

  digitalWrite(EN_ALL, HIGH);

  Serial.println("Mega Ready");
}

// ================== MOTOR FUNCTIONS ==================
void driveMotors(int leftSpeed, int rightSpeed) {
  if (leftSpeed >= 0) {
    analogWrite(LPWM_LEFT, 0);
    analogWrite(RPWM_LEFT, constrain(leftSpeed, 0, 255));
  } else {
    analogWrite(LPWM_LEFT, constrain(-leftSpeed, 0, 255));
    analogWrite(RPWM_LEFT, 0);
  }
  
  if (rightSpeed >= 0) {
    analogWrite(LPWM_RIGHT, 0);
    analogWrite(RPWM_RIGHT, constrain(rightSpeed, 0, 255));
  } else {
    analogWrite(LPWM_RIGHT, constrain(-rightSpeed, 0, 255));
    analogWrite(RPWM_RIGHT, 0);
  }
}

void stopAllMotors() {
  analogWrite(LPWM_LEFT, 0);
  analogWrite(RPWM_LEFT, 0);
  analogWrite(LPWM_RIGHT, 0);
  analogWrite(RPWM_RIGHT, 0);
  // Không tắt motor bắn ở đây - để soft stop xử lý
}

void stopShootMotor() {
  // Tắt cứng motor bắn (dùng khi cần dừng ngay)
  shootPWM = 0;
  shootPWM_Target = 0;
  analogWrite(LPWM_SHOOT, 0);
  analogWrite(RPWM_SHOOT, 0);
}

// ================== SOFT SHOOT ==================
void updateSoftShoot() {
  unsigned long now = millis();
  if (now - lastShootUpdate >= 20) {
    lastShootUpdate = now;

    if (shootPWM < shootPWM_Target) {
      shootPWM += 3;
      if (shootPWM > shootPWM_Target) shootPWM = shootPWM_Target;
    }
    else if (shootPWM > shootPWM_Target) {
      shootPWM -= 3;
      if (shootPWM < shootPWM_Target) shootPWM = shootPWM_Target;
    }

    analogWrite(LPWM_SHOOT, 0);
    analogWrite(RPWM_SHOOT, shootPWM);
  }
}

// ================== TÍNH TỐC ĐỘ THEO REGION ==================
void calculateSpeedFromRegion(int region, int &leftSpeed, int &rightSpeed) {
  int absRegion = abs(region);
  int speedDiff = absRegion * speedDiffPerRegion;
  
  if (region == 0) {
    leftSpeed = baseSpeed;
    rightSpeed = baseSpeed;
  }
  else if (region < 0) {
    leftSpeed = baseSpeed - speedDiff;
    rightSpeed = baseSpeed + speedDiff;
  }
  else {
    leftSpeed = baseSpeed + speedDiff;
    rightSpeed = baseSpeed - speedDiff;
  }
  
  leftSpeed = constrain(leftSpeed, minSpeed, maxSpeed);
  rightSpeed = constrain(rightSpeed, minSpeed, maxSpeed);
  
  Serial.print("R=");
  Serial.print(region);
  Serial.print(" L=");
  Serial.print(leftSpeed);
  Serial.print(" R=");
  Serial.println(rightSpeed);
}

// ================== NÉ NGƯỜI ==================
void avoidPerson(int region) {
  avoidingPerson = true;
  avoidStartTime = millis();
  shootPWM_Target = 0;  // Tắt motor bắn khi né
  
  if (region == 5) {
    stopAllMotors();
    Serial.println("[NE] DUNG!");
  }
  else if (region > 0) {
    driveMotors(-avoidSpeed, avoidSpeed);
    Serial.println("[NE] TRAI");
  }
  else {
    driveMotors(avoidSpeed, -avoidSpeed);
    Serial.println("[NE] PHAI");
  }
}

// ================== LOOP ==================
void loop() {
  // Luôn cập nhật soft shoot (quan trọng!)
  updateSoftShoot();

  // ----- 1. Nhận lệnh từ PC -----
  if (Serial3.available() > 0) {
    String line = Serial3.readStringUntil('\n');
    line.trim();

    if (line.length() > 0) {
      int cls = 0, region = 5;
      int parsed = sscanf(line.c_str(), "%d %d", &cls, &region);

      if (parsed == 2) {
        region = constrain(region, -4, 5);
        lastCommandTime = millis();

        // CLASS 1: NÉ NGƯỜI
        if (cls == 1) {
          avoidPerson(region);
          return;
        }

        // CLASS 0, 2: THEO TARGET
        avoidingPerson = false;
        
        if (mode == 1 && cls == 0) {
          setpoint = region;
          pcCommandReceived = true;
        }
        else if (mode == 2 && cls == 2) {
          setpoint = region;
          pcCommandReceived = true;
        }
      }
    }
  }

  // ----- 2. Đang né người? -----
  if (avoidingPerson) {
    if (millis() - avoidStartTime < avoidDuration) {
      return;
    }
    avoidingPerson = false;
    stopAllMotors();
  }

  // ----- 3. Timeout -----
  if (pcCommandReceived && (millis() - lastCommandTime > commandTimeout)) {
    pcCommandReceived = false;
    setpoint = 5;
    shootPWM_Target = 0;  // Tắt motor bắn khi timeout
    Serial.println("TIMEOUT");
  }

  // ----- 4. Đếm bóng -----
  int irReading = digitalRead(ballSensorPin);
  if (irReading != lastIrReading) {
    lastDebounceTime = millis();
  }
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (irReading != irSensorState) {
      irSensorState = irReading;
      if (mode == 1 && irSensorState == HIGH) {
        ballCount++;
        Serial.print("Ball: ");
        Serial.println(ballCount);
        if (ballCount >= 3) {
          mode = 2;
          setpoint = 5;
          pcCommandReceived = false;
          shootPWM_Target = 0;  // Tắt motor bắn khi chuyển mode
          Serial.println(">>> MODE 2");
        }
      }
    }
  }
  lastIrReading = irReading;

  // ----- 5. Nam châm -----
  int mag1 = digitalRead(magnetSensor1Pin);
  int mag2 = digitalRead(magnetSensor2Pin);
  if (mode == 2 && (mag1 == LOW || mag2 == LOW)) {
    Serial.println(">>> NAM CHAM - Do bong");
    
    // Dừng và tắt motor bắn
    stopAllMotors();
    stopShootMotor();
    
    // Đổ bóng 15 giây
    delay(15000);
    
    // Lùi ra
    Serial.println("Lui...");
    driveMotors(-baseSpeed, -baseSpeed);
    delay(1500);
    
    // Dừng
    stopAllMotors();
    delay(500);
    
    // Reset về Mode 1
    mode = 1;
    ballCount = 0;
    setpoint = 5;
    pcCommandReceived = false;
    
    // GỬI MODE:1 VỀ PC NGAY LẬP TỨC
    Serial3.println("MODE:1");
    Serial.println(">>> MODE 1 - Da gui PC");
    
    return;
  }

  // ----- 6. Gửi mode định kỳ -----
  if (millis() - lastModeReport >= 300) {
    lastModeReport = millis();
    Serial3.print("MODE:");
    Serial3.println(mode);
  }

  // ----- 7. Điều khiển motor -----
  if (!pcCommandReceived || setpoint == 5) {
    stopAllMotors();
    shootPWM_Target = 0;  // Tắt motor bắn khi dừng
    return;
  }

  int leftSpeed, rightSpeed;
  calculateSpeedFromRegion((int)setpoint, leftSpeed, rightSpeed);
  driveMotors(leftSpeed, rightSpeed);

  // Motor bắn: chỉ bật khi đi thẳng (setpoint == 0)
  if (setpoint == 0) {
    shootPWM_Target = 80;
  } else {
    shootPWM_Target = 0;
  }
}

////===============================================================
# YOLOv5 Detection + Né người + Serial Communication

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
import serial
import time

arduino_mode = 1
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow,
    check_requirements, colorstr, cv2, increment_path,
    non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# ============ CẤU HÌNH ============
SEND_INTERVAL = 0.05
last_send_time = 0

# Ngưỡng né người
PERSON_AREA_THRESHOLD = 0.15  # Người chiếm > 15% màn hình → dừng khẩn cấp
PERSON_CONF_THRESHOLD = 0.4   # Confidence tối thiểu để né

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data_train/images",
    data=ROOT / "data_train/data_train.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    global arduino_mode, last_send_time
    
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # Serial
    try:
        arduino = serial.Serial('COM7', 115200, timeout=1)
        time.sleep(2)
        print("✓ Kết nối Arduino thành công")
    except Exception as e:
        arduino = None
        print(f"✗ Lỗi Serial: {e}")

    def update_arduino_mode():
        global arduino_mode
        if arduino and arduino.in_waiting:
            try:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("MODE:"):
                    mode_val = int(line.split(":")[1])
                    if mode_val in [1, 2]:
                        arduino_mode = mode_val
                        print(f"[MODE] → {arduino_mode}")
            except Exception as e:
                print(f"Lỗi đọc mode: {e}")

    def send_to_arduino(cls_id, region, priority="normal"):
        global last_send_time
        now = time.time()
        # Lệnh khẩn cấp (né người) được gửi ngay
        if priority == "urgent" or (now - last_send_time >= SEND_INTERVAL):
            try:
                data = f"{cls_id} {region}\n"
                arduino.write(data.encode())
                last_send_time = now
                label = {0: "BONG", 1: "NE_NGUOI", 2: "KHU_CHUA"}
                print(f"[SEND] {label.get(cls_id, cls_id)} | Region={region} | {priority.upper()}")
            except Exception as e:
                print(f"Lỗi gửi: {e}")

    for path, im, im0s, vid_cap, s in dataset:
        update_arduino_mode()
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize_path).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize_path).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize_path)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            h, w = im0.shape[:2]
            frame_area = h * w
            region_width = w / 9

            # ===== BIẾN TRACKING =====
            saw_target = False
            best_target_conf = 0
            best_target_region = 5
            best_target_cls = -1
            
            # Biến phát hiện người
            person_detected = False
            person_too_close = False
            person_region = 0
            person_max_area = 0

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    confidence = float(conf)
                    
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    center_x = (x1 + x2) / 2
                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_ratio = bbox_area / frame_area
                    
                    region_index = int((center_x / region_width) - 4)
                    region_index = max(-4, min(4, region_index))

                    # ===== XỬ LÝ NGƯỜI (CLASS 1) =====
                    if c == 1 and confidence >= PERSON_CONF_THRESHOLD: # Người tối thiểu để né
                        person_detected = True
                        
                        # Kiểm tra người có quá gần không
                        if area_ratio > PERSON_AREA_THRESHOLD: # giới hạn ngươfi quá gần
                            person_too_close = True
                        
                        # Lưu vùng của người lớn nhất
                        if area_ratio > person_max_area:
                            person_max_area = area_ratio
                            person_region = region_index
                        
                        # Vẽ bbox người màu đỏ
                        label = f"NGUOI {conf:.2f} R:{region_index} A:{area_ratio*100:.0f}%"
                        annotator.box_label(xyxy, label, color=(0, 0, 255))
                    
                    # ===== XỬ LÝ TARGET (BÓNG / KHU CHỨA) =====
                    elif (arduino_mode == 1 and c == 0) or (arduino_mode == 2 and c == 2): # Bóng hoặc khu chứa
                        # Cập nhật target tốt nhất
                        if confidence > best_target_conf:
                            best_target_conf = confidence
                            best_target_region = region_index
                            best_target_cls = c
                            saw_target = True
                        
                        label = f"{names[c]} {conf:.2f} R:{region_index}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    else:
                        # Vẽ các class khác
                        if view_img:
                            label = f"{names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))

            # ===== QUYẾT ĐỊNH GỬI LỆNH =====
            if arduino:
                if person_detected:
                    if person_too_close:
                        # NGƯỜI QUÁ GẦN → DỪNG KHẨN CẤP
                        send_to_arduino(1, 5, "urgent")
                        status = "🛑 DUNG KHAN CAP - NGUOI QUA GAN!"
                    elif person_region <= 0:
                        # Người bên trái/giữa → né sang phải
                        send_to_arduino(1, 3, "urgent")
                        status = "↪️ NE PHAI (nguoi ben trai)"
                    else:
                        # Người bên phải → né sang trái
                        send_to_arduino(1, -3, "urgent")
                        status = "↩️ NE TRAI (nguoi ben phai)"
                
                elif saw_target:
                    # Không có người, theo target bình thường
                    send_to_arduino(best_target_cls, best_target_region, "normal")
                    status = f"🎯 Theo target cls={best_target_cls} reg={best_target_region}"
                
                elif arduino_mode == 1:
                    # Mode 1, không thấy gì → quay tìm
                    send_to_arduino(0, 2, "normal")
                    status = "🔄 Tim bong..."
                
                else:
                    # Mode 2, không thấy khu → dừng chờ
                    send_to_arduino(2, 5, "normal")
                    status = "⏸️ Cho tim khu chua bong..."

            # ===== VẼ THÔNG TIN LÊN FRAME =====
            # Chia vùng
            step = w // 9
            for j in range(1, 9):
                x = int(step * j)
                cv2.line(im0, (x, 0), (x, h), (0, 255, 255), 1)
            for j in range(9):
                text = str(j - 4)
                x_pos = int(step * (j + 0.5))
                cv2.putText(im0, text, (x_pos - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Thông tin
            cv2.putText(im0, f"MODE: {arduino_mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Cảnh báo người
            if person_detected:
                color = (0, 0, 255) if person_too_close else (0, 165, 255)
                warn_text = "!! NGUOI QUA GAN !!" if person_too_close else f"Nguoi: vung {person_region}"
                cv2.putText(im0, warn_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if 'status' in locals():
                cv2.putText(im0, status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    break

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    while len(vid_path) <= i:
                        vid_path.append('')
                    while len(vid_writer) <= i:
                        vid_writer.append(None)
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w_vid = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h_vid = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w_vid, h_vid = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true", default=True)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-format", type=int, default=0)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dnn", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
