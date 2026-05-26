import os
import sys
import argparse
import glob
import time
import csv
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
# KONFIGURASI KHUSUS CCTV SPBU
# Dioptimalkan untuk:
# - Jarak kamera ke objek : ±5 meter
# - Sudut kamera          : dari atas (bird-eye view)
# - Kondisi cahaya        : siang hari outdoor
# - Objek target          : jerigen di SPBU
# ─────────────────────────────────────────────
CCTV_CONFIG = {
    # Confidence lebih rendah karena jerigen tampak kecil dari 5 meter
    "conf_threshold"    : 0.25,

    # Ukuran minimum bounding box jerigen (pixel)
    # Jerigen dari jarak 5m di resolusi 640x480 kira-kira 40x60 pixel
    "min_box_lebar"     : 30,
    "min_box_tinggi"    : 30,

    # Ukuran maksimum bounding box (filter objek terlalu besar)
    "max_box_lebar"     : 300,
    "max_box_tinggi"    : 300,

    # Ukuran gambar inferensi — naikkan ke 1280 untuk objek kecil dari jauh
    "imgsz"             : 1280,

    # Nama kelas yang ingin dideteksi (sesuaikan dengan label training Anda)
    "kelas_target"      : ["jerigen"],

    # CSV logging
    "csv_file"          : "deteksi_jerigen_cctv.csv",
    "interval_catat"    : 30,   # catat setiap 30 frame

    # Simpan screenshot otomatis saat jerigen terdeteksi
    "auto_screenshot"   : True,
    "folder_screenshot" : "screenshot_jerigen",

    # Zona deteksi (ROI = Region of Interest)
    # Set True jika ingin batasi area deteksi di layar
    # Berguna untuk fokus ke area pengisian BBM saja
    "gunakan_roi"       : False,
    "roi_x1"            : 100,   # koordinat kiri
    "roi_y1"            : 100,   # koordinat atas
    "roi_x2"            : 700,   # koordinat kanan
    "roi_y2"            : 600,   # koordinat bawah
}

# ─────────────────────────────────────────────
# ARGUMEN INPUT
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',
    help='Path ke model YOLO (contoh: "best.pt")',
    required=True)
parser.add_argument('--source',
    help='Sumber video/gambar (contoh: "cctv.mp4", "usb0")',
    required=True)
parser.add_argument('--thresh',
    help='Confidence threshold (default: 0.25)',
    default=0.25)
parser.add_argument('--resolution',
    help='Resolusi tampilan WxH (contoh: "1280x720")',
    default=None)
parser.add_argument('--record',
    help='Rekam hasil deteksi ke file demo1.avi',
    action='store_true')
parser.add_argument('--roi',
    help='Aktifkan Region of Interest (area deteksi terbatas)',
    action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res   = args.resolution
record     = args.record

# Override ROI dari argumen jika diberikan
if args.roi:
    CCTV_CONFIG["gunakan_roi"] = True

# ─────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────

def inisialisasi_csv(nama_file):
    """Buat file CSV dengan header."""
    if not os.path.exists(nama_file):
        with open(nama_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "No", "Tanggal", "Jam", "Nama_Objek",
                "Confidence (%)", "Jumlah_Terdeteksi",
                "Waktu_Video", "Lebar_Box (px)", "Tinggi_Box (px)",
                "Posisi_X", "Posisi_Y", "Sumber_Video",
            ])
        print(f"✅ File CSV dibuat: {nama_file}")


def catat_csv(nama_file, no, nama_obj, conf, jumlah,
              waktu_vid, lebar, tinggi, cx, cy, sumber):
    """Tulis satu baris deteksi ke CSV."""
    now = datetime.now()
    with open(nama_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            no,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            nama_obj,
            f"{conf:.1f}",
            jumlah,
            waktu_vid,
            lebar,
            tinggi,
            cx,
            cy,
            os.path.basename(sumber),
        ])


def format_waktu(detik):
    """Ubah detik ke HH:MM:SS."""
    if detik is None:
        return "-"
    td = timedelta(seconds=int(detik))
    h, r = divmod(td.seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def preprocess_cctv(frame):
    """
    Preprocessing khusus untuk gambar CCTV dari jarak 5 meter.
    Meningkatkan kontras dan ketajaman agar jerigen lebih mudah terdeteksi.
    """
    # 1. Tingkatkan kontras dengan CLAHE
    #    (sangat membantu untuk CCTV dengan cahaya tidak merata)
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Sedikit sharpen agar tepi jerigen lebih jelas
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    frame = cv2.filter2D(frame, -1, kernel)

    return frame


def gambar_roi(frame, x1, y1, x2, y2):
    """Gambar kotak ROI di layar sebagai referensi area deteksi."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(overlay, "ZONA DETEKSI", (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


def filter_box_ukuran(xmin, ymin, xmax, ymax):
    """
    Filter bounding box berdasarkan ukuran.
    Jerigen dari jarak 5 meter tidak terlalu besar maupun kecil.
    """
    lebar  = xmax - xmin
    tinggi = ymax - ymin

    terlalu_kecil = (lebar < CCTV_CONFIG["min_box_lebar"] or
                     tinggi < CCTV_CONFIG["min_box_tinggi"])
    terlalu_besar = (lebar > CCTV_CONFIG["max_box_lebar"] or
                     tinggi > CCTV_CONFIG["max_box_tinggi"])

    return not (terlalu_kecil or terlalu_besar)


def dalam_roi(cx, cy):
    """Cek apakah pusat objek berada di dalam zona ROI."""
    if not CCTV_CONFIG["gunakan_roi"]:
        return True
    return (CCTV_CONFIG["roi_x1"] <= cx <= CCTV_CONFIG["roi_x2"] and
            CCTV_CONFIG["roi_y1"] <= cy <= CCTV_CONFIG["roi_y2"])


# ─────────────────────────────────────────────
# CEK FILE MODEL
# ─────────────────────────────────────────────
if not os.path.exists(model_path):
    print(f'❌ ERROR: Model tidak ditemukan di: {model_path}')
    sys.exit(0)

# Load model
model  = YOLO(model_path, task='detect')
labels = model.names
print(f"✅ Model dimuat: {model_path}")
print(f"   Kelas tersedia: {list(labels.values())}\n")

# ─────────────────────────────────────────────
# TENTUKAN TIPE SUMBER INPUT
# ─────────────────────────────────────────────
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'❌ Format file {ext} tidak didukung.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'❌ Sumber input tidak valid: {img_source}')
    sys.exit(0)

# ─────────────────────────────────────────────
# SETUP RESOLUSI DAN RECORDING
# ─────────────────────────────────────────────
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

if record:
    if source_type not in ['video', 'usb']:
        print('❌ Recording hanya untuk video/kamera.')
        sys.exit(0)
    if not user_res:
        print('❌ Tentukan resolusi dengan --resolution untuk recording.')
        sys.exit(0)
    recorder = cv2.VideoWriter(
        'demo1.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30, (resW, resH)
    )

# ─────────────────────────────────────────────
# LOAD SUMBER INPUT
# ─────────────────────────────────────────────
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        _, ext = os.path.splitext(file)
        if ext in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# ─────────────────────────────────────────────
# INISIALISASI CSV & FOLDER SCREENSHOT
# ─────────────────────────────────────────────
inisialisasi_csv(CCTV_CONFIG["csv_file"])

if CCTV_CONFIG["auto_screenshot"]:
    os.makedirs(CCTV_CONFIG["folder_screenshot"], exist_ok=True)

# Warna bounding box per kelas
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168), (169,162,241),
    (98,118,150), (172,176,184)
]

# Variabel kontrol
avg_frame_rate   = 0
frame_rate_buffer = []
fps_avg_len      = 200
img_count        = 0
frame_ke         = 0
no_csv           = 1
frame_dicatat    = -999
screenshot_count = 0

print("═"*55)
print("  🎯 DETEKSI JERIGEN CCTV SPBU — YOLOv8")
print(f"  Confidence threshold : {min_thresh}")
print(f"  Ukuran inferensi     : {CCTV_CONFIG['imgsz']}px")
print(f"  ROI aktif            : {CCTV_CONFIG['gunakan_roi']}")
print(f"  Auto screenshot      : {CCTV_CONFIG['auto_screenshot']}")
print("  Tekan 'Q' untuk berhenti")
print("  Tekan 'P' untuk screenshot manual")
print("  Tekan 'R' untuk aktifkan/nonaktifkan ROI")
print("═"*55 + "\n")

# ─────────────────────────────────────────────
# LOOP INFERENSI UTAMA
# ─────────────────────────────────────────────
while True:

    t_start  = time.perf_counter()
    frame_ke += 1

    # ── Load frame ──────────────────────────
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('✅ Semua gambar selesai diproses.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('✅ Video selesai diproses.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if frame is None or not ret:
            print('❌ Kamera tidak terbaca.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('❌ Picamera tidak terbaca.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ── Preprocessing khusus CCTV ───────────
    frame_processed = preprocess_cctv(frame.copy())

    # ── Gambar zona ROI jika aktif ───────────
    if CCTV_CONFIG["gunakan_roi"]:
        frame = gambar_roi(
            frame,
            CCTV_CONFIG["roi_x1"], CCTV_CONFIG["roi_y1"],
            CCTV_CONFIG["roi_x2"], CCTV_CONFIG["roi_y2"]
        )

    # ── Inferensi YOLOv8 ────────────────────
    results = model(
        frame_processed,
        verbose = False,
        imgsz   = CCTV_CONFIG["imgsz"],
        conf    = min_thresh,
    )

    detections   = results[0].boxes
    object_count = 0
    deteksi_frame = {}

    # ── Proses setiap deteksi ───────────────
    for i in range(len(detections)):

        xyxy   = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx  = int(detections[i].cls.item())
        classname = labels[classidx]
        conf      = detections[i].conf.item()

        lebar  = xmax - xmin
        tinggi = ymax - ymin
        cx     = (xmin + xmax) // 2
        cy     = (ymin + ymax) // 2

        # Filter 1: confidence threshold
        if conf < min_thresh:
            continue

        # Filter 2: ukuran bounding box
        if not filter_box_ukuran(xmin, ymin, xmax, ymax):
            continue

        # Filter 3: dalam zona ROI
        if not dalam_roi(cx, cy):
            continue

        # Filter 4: hanya kelas target
        nama_lower = classname.lower()
        is_target  = any(
            k.lower() in nama_lower
            for k in CCTV_CONFIG["kelas_target"]
        )

        # Gambar bounding box
        color = bbox_colors[classidx % 10]
        tebal = 3 if is_target else 1  # lebih tebal untuk kelas target

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, tebal)

        # Label dengan confidence
        label      = f'{classname}: {int(conf*100)}%'
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10)

        cv2.rectangle(frame,
            (xmin, label_ymin - labelSize[1] - 10),
            (xmin + labelSize[0], label_ymin + baseLine - 10),
            color, cv2.FILLED)
        cv2.putText(frame, label,
            (xmin, label_ymin - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Tandai pusat objek
        cv2.circle(frame, (cx, cy), 4, color, -1)

        object_count += 1

        # Kumpulkan untuk CSV (hanya kelas target)
        if is_target:
            if classname not in deteksi_frame:
                deteksi_frame[classname] = []
            deteksi_frame[classname].append({
                "conf"  : conf * 100,
                "lebar" : lebar,
                "tinggi": tinggi,
                "cx"    : cx,
                "cy"    : cy,
            })

    # ── Screenshot otomatis saat jerigen terdeteksi ──
    if deteksi_frame and CCTV_CONFIG["auto_screenshot"]:
        if (frame_ke - frame_dicatat) >= CCTV_CONFIG["interval_catat"]:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_ss = f"{CCTV_CONFIG['folder_screenshot']}/jerigen_{ts}_{screenshot_count:04d}.jpg"
            cv2.imwrite(path_ss, frame)
            screenshot_count += 1

    # ── Catat ke CSV ────────────────────────
    if deteksi_frame and (frame_ke - frame_dicatat) >= CCTV_CONFIG["interval_catat"]:

        waktu_vid = None
        if source_type == 'video':
            waktu_vid = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for nama_kelas, data_list in deteksi_frame.items():
            conf_rata = sum(d["conf"] for d in data_list) / len(data_list)
            jumlah    = len(data_list)
            lebar_avg = int(sum(d["lebar"] for d in data_list) / jumlah)
            tinggi_avg= int(sum(d["tinggi"] for d in data_list) / jumlah)
            cx_avg    = int(sum(d["cx"] for d in data_list) / jumlah)
            cy_avg    = int(sum(d["cy"] for d in data_list) / jumlah)

            catat_csv(
                CCTV_CONFIG["csv_file"],
                no_csv, nama_kelas, conf_rata, jumlah,
                format_waktu(waktu_vid),
                lebar_avg, tinggi_avg, cx_avg, cy_avg,
                img_source,
            )

            print(f"📝 [{no_csv}] {nama_kelas} | {jumlah} objek | "
                  f"conf: {conf_rata:.1f}% | "
                  f"box: {lebar_avg}x{tinggi_avg}px | "
                  f"waktu: {format_waktu(waktu_vid)}")

            no_csv += 1

        frame_dicatat = frame_ke

    # ── Overlay info di layar ────────────────
    # Background info bar
    cv2.rectangle(frame, (0, 0), (380, 75), (0, 0, 0), cv2.FILLED)

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.1f}',
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f'Jerigen terdeteksi: {object_count}',
        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f'Total dicatat: {no_csv - 1}',
        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Indikator DETEKSI saat jerigen ditemukan
    if deteksi_frame:
        cv2.rectangle(frame,
            (frame.shape[1]-200, 0),
            (frame.shape[1], 35), (0, 0, 200), cv2.FILLED)
        cv2.putText(frame, '⚠ JERIGEN!',
            (frame.shape[1]-195, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Deteksi Jerigen CCTV SPBU — YOLOv8', frame)
    if record:
        recorder.write(frame)

    # ── Kontrol keyboard ────────────────────
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Dihentikan oleh pengguna.")
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'capture_{ts}.jpg', frame)
        print(f"📸 Screenshot disimpan: capture_{ts}.jpg")
    elif key == ord('r') or key == ord('R'):
        # Toggle ROI on/off
        CCTV_CONFIG["gunakan_roi"] = not CCTV_CONFIG["gunakan_roi"]
        status = "ON" if CCTV_CONFIG["gunakan_roi"] else "OFF"
        print(f"🔲 ROI diubah: {status}")

    # ── Hitung FPS ──────────────────────────
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)


# ─────────────────────────────────────────────
# SELESAI — LAPORAN AKHIR
# ─────────────────────────────────────────────
print(f'\n{"═"*55}')
print(f'  ✅ DETEKSI SELESAI')
print(f'  Rata-rata FPS      : {avg_frame_rate:.2f}')
print(f'  Total frame        : {frame_ke}')
print(f'  Total data CSV     : {no_csv - 1} baris')
print(f'  File CSV           : {CCTV_CONFIG["csv_file"]}')
print(f'  Screenshot tersimpan: {screenshot_count} gambar')
print(f'{"═"*55}')

if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
