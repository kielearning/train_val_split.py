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
# - Mode deteksi          : SEGMENTASI (mask/polygon)
# ─────────────────────────────────────────────
CCTV_CONFIG = {
    # Confidence lebih rendah karena jerigen tampak kecil dari 5 meter
    "conf_threshold"    : 0.25,

    # Ukuran minimum bounding box jerigen (pixel)
    # Jerigen dari jarak 5m di resolusi 640x480 kira-kira 30x30 pixel
    "min_box_lebar"     : 30,
    "min_box_tinggi"    : 30,

    # Ukuran maksimum bounding box
    "max_box_lebar"     : 400,
    "max_box_tinggi"    : 400,

    # Ukuran gambar inferensi — 1280 untuk objek kecil dari jauh
    "imgsz"             : 1280,

    # Nama kelas yang ingin dideteksi
    "kelas_target"      : ["jerigen"],

    # CSV logging
    "csv_file"          : "deteksi_jerigen_cctv.csv",
    "interval_catat"    : 30,   # catat setiap 30 frame (±1 detik di 30fps)

    # Screenshot otomatis saat jerigen terdeteksi
    "auto_screenshot"   : True,
    "folder_screenshot" : "screenshot_jerigen",

    # Zona ROI (Region of Interest)
    # Aktifkan jika ingin batasi area deteksi ke zona pompa BBM saja
    "gunakan_roi"       : False,
    "roi_x1"            : 100,
    "roi_y1"            : 100,
    "roi_x2"            : 700,
    "roi_y2"            : 600,

    # Transparansi mask segmentasi (0.0 = transparan, 1.0 = solid)
    "mask_alpha"        : 0.4,
}

# ─────────────────────────────────────────────
# ARGUMEN INPUT
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',
    help='Path ke model YOLO segmentasi (contoh: "runs/segment/train/weights/best.pt")',
    required=True)
parser.add_argument('--source',
    help='Sumber input: file gambar, folder, video, "usb0", atau "picamera0"',
    required=True)
parser.add_argument('--thresh',
    help='Confidence threshold minimum (contoh: "0.25")',
    default=0.25)
parser.add_argument('--resolution',
    help='Resolusi tampilan WxH (contoh: "1280x720")',
    default=None)
parser.add_argument('--record',
    help='Rekam hasil ke demo1.avi. Wajib tentukan --resolution.',
    action='store_true')
parser.add_argument('--roi',
    help='Aktifkan zona ROI untuk batasi area deteksi',
    action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res   = args.resolution
record     = args.record

if args.roi:
    CCTV_CONFIG["gunakan_roi"] = True

# ─────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────

def inisialisasi_csv(nama_file):
    """Membuat file CSV dengan header jika belum ada."""
    if not os.path.exists(nama_file):
        with open(nama_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "No",
                "Tanggal",
                "Jam",
                "Nama_Objek",
                "Confidence (%)",
                "Jumlah_Terdeteksi",
                "Waktu_Video",
                "Luas_Mask (px²)",
                "Posisi_X (center)",
                "Posisi_Y (center)",
                "Sumber_Video",
            ])
        print(f"✅ File CSV dibuat: {nama_file}")
    else:
        print(f"✅ File CSV sudah ada, data akan ditambahkan: {nama_file}")


def catat_csv(nama_file, no, nama_obj, conf, jumlah,
              waktu_vid, luas_mask, cx, cy, sumber):
    """Menulis satu baris deteksi ke file CSV."""
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
            luas_mask,
            cx,
            cy,
            os.path.basename(sumber),
        ])


def format_waktu(detik):
    """Mengubah detik menjadi format HH:MM:SS."""
    if detik is None:
        return "-"
    td = timedelta(seconds=int(detik))
    h, r = divmod(td.seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def preprocess_cctv(frame):
    """
    Preprocessing khusus CCTV jarak 5 meter.
    Meningkatkan kontras dan ketajaman agar jerigen lebih mudah terdeteksi.
    """
    # CLAHE: tingkatkan kontras secara lokal
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen: perjelas tepi objek yang blur karena jarak
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame


def gambar_roi(frame):
    """Menggambar zona ROI di layar sebagai referensi area deteksi."""
    x1 = CCTV_CONFIG["roi_x1"]
    y1 = CCTV_CONFIG["roi_y1"]
    x2 = CCTV_CONFIG["roi_x2"]
    y2 = CCTV_CONFIG["roi_y2"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "ZONA DETEKSI",
                (x1 + 5, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def filter_ukuran_polygon(polygon):
    """
    Filter polygon mask berdasarkan ukuran bounding box-nya.
    Menyaring objek terlalu kecil atau terlalu besar.
    """
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    lebar  = x_coords.max() - x_coords.min()
    tinggi = y_coords.max() - y_coords.min()

    terlalu_kecil = (lebar < CCTV_CONFIG["min_box_lebar"] or
                     tinggi < CCTV_CONFIG["min_box_tinggi"])
    terlalu_besar = (lebar > CCTV_CONFIG["max_box_lebar"] or
                     tinggi > CCTV_CONFIG["max_box_tinggi"])

    return not (terlalu_kecil or terlalu_besar)


def dalam_roi(cx, cy):
    """Mengecek apakah pusat objek berada di dalam zona ROI."""
    if not CCTV_CONFIG["gunakan_roi"]:
        return True
    return (CCTV_CONFIG["roi_x1"] <= cx <= CCTV_CONFIG["roi_x2"] and
            CCTV_CONFIG["roi_y1"] <= cy <= CCTV_CONFIG["roi_y2"])


def hitung_luas_polygon(polygon):
    """Menghitung luas area mask polygon dalam satuan pixel²."""
    return cv2.contourArea(polygon)


# ─────────────────────────────────────────────
# CEK DAN LOAD MODEL
# ─────────────────────────────────────────────
if not os.path.exists(model_path):
    print(f'❌ ERROR: Model tidak ditemukan di: {model_path}')
    sys.exit(0)

model  = YOLO(model_path, task='segment')
labels = model.names
print(f"✅ Model segmentasi dimuat: {model_path}")
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
        print('❌ Recording hanya untuk sumber video/kamera.')
        sys.exit(0)
    if not user_res:
        print('❌ Tentukan --resolution untuk mengaktifkan recording.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps  = 30
    recorder    = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        record_fps, (resW, resH)
    )

# ─────────────────────────────────────────────
# LOAD SUMBER INPUT
# ─────────────────────────────────────────────
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
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

# Warna mask per kelas (Tableau 10)
mask_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
    (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
]

# Variabel kontrol
avg_frame_rate    = 0
frame_rate_buffer = []
fps_avg_len       = 200
img_count         = 0
frame_ke          = 0
no_csv            = 1
frame_dicatat     = -999
screenshot_count  = 0
mask_alpha        = CCTV_CONFIG["mask_alpha"]

print("═"*58)
print("  🎯 DETEKSI JERIGEN CCTV SPBU — YOLOv8 SEGMENTASI")
print(f"  Confidence threshold : {min_thresh}")
print(f"  Ukuran inferensi     : {CCTV_CONFIG['imgsz']}px")
print(f"  ROI aktif            : {CCTV_CONFIG['gunakan_roi']}")
print(f"  Auto screenshot      : {CCTV_CONFIG['auto_screenshot']}")
print(f"  Output CSV           : {CCTV_CONFIG['csv_file']}")
print("─"*58)
print("  Kontrol keyboard:")
print("  Q = Keluar | P = Screenshot | S = Pause | R = Toggle ROI")
print("═"*58 + "\n")

# ─────────────────────────────────────────────
# LOOP INFERENSI UTAMA
# ─────────────────────────────────────────────
while True:

    t_start  = time.perf_counter()
    frame_ke += 1

    # ── Load frame dari sumber ───────────────
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('✅ Semua gambar selesai diproses.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
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
        frame = gambar_roi(frame)

    # ── Inferensi YOLOv8 Segmentasi ─────────
    results = model(
        frame_processed,
        verbose      = False,
        retina_masks = True,
        imgsz        = CCTV_CONFIG["imgsz"],
        conf         = min_thresh,
    )

    detections   = results[0].boxes
    masks        = results[0].masks
    object_count = 0
    deteksi_frame = {}

    # Overlay untuk efek transparansi mask
    overlay = frame.copy()

    # ── Proses mask segmentasi ───────────────
    if masks is not None:
        mask_polygons = masks.xy

        for i in range(len(detections)):

            classidx  = int(detections[i].cls.item())
            classname = labels[classidx]
            conf      = detections[i].conf.item()

            if conf < min_thresh:
                continue

            color   = mask_colors[classidx % 10]
            polygon = mask_polygons[i]

            if polygon is None or len(polygon) == 0:
                continue
            polygon = polygon.astype(np.int32)

            # Filter ukuran polygon
            if not filter_ukuran_polygon(polygon):
                continue

            # Hitung pusat objek dari polygon
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]
            cx = int((x_coords.min() + x_coords.max()) / 2)
            cy = int((y_coords.min() + y_coords.max()) / 2)

            # Filter zona ROI
            if not dalam_roi(cx, cy):
                continue

            # ── Gambar mask (sama seperti kode asli) ──
            # Isi polygon transparan di overlay
            cv2.fillPoly(overlay, [polygon], color)

            # Garis tepi polygon — lebih tebal untuk kelas target
            nama_lower = classname.lower()
            is_target  = any(
                k.lower() in nama_lower
                for k in CCTV_CONFIG["kelas_target"]
            )
            tebal_garis = 3 if is_target else 2
            cv2.polylines(frame, [polygon],
                          isClosed=True, color=color, thickness=tebal_garis)

            # ── Label di tengah objek ──
            label      = f'{classname}: {int(conf*100)}%'
            font_scale = 0.5
            font_thick = 1
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            text_x = cx - labelSize[0] // 2
            text_y = cy + labelSize[1] // 2

            cv2.rectangle(frame,
                (text_x - 2, text_y - labelSize[1] - 2),
                (text_x + labelSize[0] + 2, text_y + baseLine),
                color, cv2.FILLED)
            cv2.putText(frame, label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick)

            # Tandai titik pusat objek
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

            object_count += 1

            # ── Kumpulkan data untuk CSV (kelas target saja) ──
            if is_target:
                luas = int(hitung_luas_polygon(polygon))
                if classname not in deteksi_frame:
                    deteksi_frame[classname] = []
                deteksi_frame[classname].append({
                    "conf": conf * 100,
                    "luas": luas,
                    "cx"  : cx,
                    "cy"  : cy,
                })

    # ── Blending overlay mask + frame asli ──
    cv2.rectangle(overlay, (0, 0), (320, 80), (0, 0, 0), cv2.FILLED)
    frame = cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0)

    # ── Screenshot otomatis ─────────────────
    if deteksi_frame and CCTV_CONFIG["auto_screenshot"]:
        if (frame_ke - frame_dicatat) >= CCTV_CONFIG["interval_catat"]:
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_ss = (f"{CCTV_CONFIG['folder_screenshot']}/"
                       f"jerigen_{ts}_{screenshot_count:04d}.jpg")
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
            luas_rata = int(sum(d["luas"] for d in data_list) / jumlah)
            cx_rata   = int(sum(d["cx"]   for d in data_list) / jumlah)
            cy_rata   = int(sum(d["cy"]   for d in data_list) / jumlah)

            catat_csv(
                CCTV_CONFIG["csv_file"],
                no_csv, nama_kelas, conf_rata, jumlah,
                format_waktu(waktu_vid),
                luas_rata, cx_rata, cy_rata,
                img_source,
            )

            print(f"📝 [{no_csv}] {nama_kelas} | "
                  f"{jumlah} objek | "
                  f"conf: {conf_rata:.1f}% | "
                  f"luas mask: {luas_rata}px² | "
                  f"waktu: {format_waktu(waktu_vid)}")

            no_csv += 1

        frame_dicatat = frame_ke

    # ── Info di layar ───────────────────────
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.1f}',
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (54, 224, 54), 2)

    cv2.putText(frame, f'Jerigen: {object_count}',
        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (54, 224, 54), 2)

    cv2.putText(frame, f'Dicatat: {no_csv - 1}',
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (54, 224, 54), 2)

    # Indikator merah saat jerigen terdeteksi
    if deteksi_frame:
        lbr = frame.shape[1]
        cv2.rectangle(frame, (lbr - 210, 0), (lbr, 38), (0, 0, 180), cv2.FILLED)
        cv2.putText(frame, '⚠ JERIGEN TERDETEKSI',
            (lbr - 205, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow('Deteksi Jerigen CCTV SPBU — YOLOv8 Segmentasi', frame)
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
        nama_ss = f'capture_{ts}.jpg'
        cv2.imwrite(nama_ss, frame)
        print(f"📸 Screenshot disimpan: {nama_ss}")
    elif key == ord('r') or key == ord('R'):
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
# LAPORAN AKHIR
# ─────────────────────────────────────────────
print(f'\n{"═"*58}')
print(f'  ✅ DETEKSI SELESAI')
print(f'  Rata-rata FPS        : {avg_frame_rate:.2f}')
print(f'  Total frame diproses : {frame_ke}')
print(f'  Total data CSV       : {no_csv - 1} baris')
print(f'  File CSV             : {CCTV_CONFIG["csv_file"]}')
print(f'  Screenshot tersimpan : {screenshot_count} gambar')
print(f'{"═"*58}')

if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
