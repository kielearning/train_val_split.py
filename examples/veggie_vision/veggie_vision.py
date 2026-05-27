"""VeggieVision — estimate the volume and weight of vegetables/fruits under a USB camera
using a YOLO segmentation model.

Run: python veggie_vision.py
All configuration lives in the constants block below (no CLI args).
"""

import math
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================================================
# Configuration
# =========================================================================================

MODEL_PATH    = 'food-1-model-3.pt'   # Path to the trained YOLO segmentation model
CAMERA_INDEX  = 0                  # USB camera index (0 is usually the default webcam)
CAMERA_W      = 1280               # Requested camera width
CAMERA_H      = 720                # Requested camera height
PANEL_W       = 300                # Width of the info panel on the right side of the window
CONF_THRESH   = 0.4                # Minimum detection confidence to consider
TEST_VIDEO_NAME = 'VeggieVision-Test.avi' # Test video (if not running on USB camera)


# -----------------------------------------------------------------------------------------
# CALIBRATION: measure a known object (e.g., a ruler) under the camera once and update this.
# For example, if a 100 mm ruler spans 200 pixels in the camera feed, PIXELS_PER_MM = 2.0 .
# The camera must stay at the same height for this constant to remain valid.
# -----------------------------------------------------------------------------------------
PIXELS_PER_MM = 373/305


# Classes we assume are roughly spherical — volume = (4/3)*pi*(d/2)^3
SPHERICAL_CLASSES = {'tomato', 'yellow_onion', 'red_onion', 'cabbage', 'apple'}
# Classes we assume are oblong — volume from the disk method
OBLONG_CLASSES    = {'carrot', 'cucumber', 'potato'}

# Approximate food densities in g/mL (looked up from food-science references).
# Tweak these for your specific varieties if you want more accurate weight estimates.
DENSITIES = {
    'carrot':       1.04,
    'yellow_onion': 0.96,
    'red_onion':    0.96,
    'potato':       1.08,
    'tomato':       0.95,
    'apple':        0.75,
    'cucumber':     0.96,
    'cabbage':      0.45,
}

# Finagling Factors for spherical classes — multiplied into the volume estimate to
# correct for the fact that real vegetables aren't perfect spheres (e.g., onions are
# typically more oblate / squat than a true sphere). Look up reasonable values for
# your produce and update as needed; 1.0 means "treat as a perfect sphere".
FINAGLING_FACTORS = {
    'tomato':       1.0,
    'yellow_onion': 1.0,
    'red_onion':    1.0,
    'cabbage':      1.0,
    'apple':        1.0,
}

# Metric <-> imperial conversions
ML_PER_CUP = 236.588
G_PER_LB   = 453.592

# Tableau-10 colors used for per-class mask coloring
MASK_COLORS = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
MASK_ALPHA = 0.4  # Transparency for the mask fill overlay


# =========================================================================================
# Volume estimation
# =========================================================================================

def find_spherical_length_and_volume(polygon, pixels_per_mm):
    """Estimate the volume and diameter of a roughly-spherical object from its segmentation polygon.
    Compute the "area-equivalent radius": the radius of the circle that has the same area as the mask.

    Returns (length_mm, volume_ml) where length_mm is the area-equivalent diameter.
    """
    if len(polygon) < 3:
        return 0.0, 0.0

    area_px = cv2.contourArea(polygon.astype(np.float32))
    if area_px <= 0:
        return 0.0, 0.0

    area_mm2 = area_px / (pixels_per_mm ** 2)
    r_mm = math.sqrt(area_mm2 / math.pi)           # A = pi*r^2  --->  r = sqrt(A/pi)
    vol_mm3 = (4.0 / 3.0) * math.pi * r_mm ** 3    # V=(4/3)*pi*r^3
    return 2.0 * r_mm, vol_mm3 / 1000.0  # diameter in mm,  mm^3 -> mL


def find_oblong_length_and_volume(polygon, pixels_per_mm):
    """Estimate the volume and long-axis length of an oblong object using the disk method from calculus.

    Steps:
    1. Rasterize the mask polygon into a local binary image.
    2. Find its major-axis orientation via PCA on the polygon points.
    3. Rotate the mask so the major axis is horizontal.
    4. For each column of the rotated mask, the non-zero pixel count is the diameter
       of the object at that slice of the axis.
    5. Volume = sum(pi * r^2 * dx) across all columns, in mm.
    6. Long-axis length = number of columns containing at least one mask pixel.

    Returns (length_mm, volume_ml).
    """
    if len(polygon) < 5:
        return 0.0, 0.0

    pts = polygon.astype(np.float32)

    # --- 1. Rasterize the mask polygon into a local binary image

    # Copy the polygon into its own array and shift it to the top-left corner
    pts_local = pts.copy()
    pts_local[:, 0] -= pts_local[:, 0].min() # Shift the polygon all the way to the left
    pts_local[:, 1] -= pts_local[:, 1].min() # Shift the polygon all the way upward

    # Find the width and height of the polygon bounding box
    w = int(math.ceil(pts_local[:, 0].max())) + 1 # Right-most edge of polygon
    h = int(math.ceil(pts_local[:, 1].max())) + 1 # Bottom-most edge of polygon
    if w < 5 or h < 5: # If polygon is too small, skip it
        return 0.0, 0.0

    # Create an new image that only contains the polygon, all points inside the polygon are filled with 255 pixel value
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_local.astype(np.int32)], 255)

    # --- 2. Use Principal Component Analysis to find the major axis (i.e. the long axis) and rotation of the polygon
    mean, unit_vector = cv2.PCACompute(pts, mean=None, maxComponents=1) # Returns a unit_vector that gives the angle of the axis
    angle_deg = math.degrees(math.atan2(unit_vector[0, 1], unit_vector[0, 0])) # Calculate rotation angle from unit_vctor

    # --- 3. Rotate the mask so the major axis is horizontal
    # Calculate matrix for rotating axis to 0°
    center = (w / 2.0, h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0) 

    # Need to expand the output canvas so we don't clip corners during rotation
    cos_a = abs(M_rot[0, 0]) # |cos(angle)|
    sin_a = abs(M_rot[0, 1]) # |sin(angle)|
    new_w = int(h * sin_a + w * cos_a) + 1 # Calculate new width
    new_h = int(h * cos_a + w * sin_a) + 1 # Calculate new height
    M_rot[0, 2] += (new_w / 2.0) - center[0] # Calculate new center x for rotated polygon
    M_rot[1, 2] += (new_h / 2.0) - center[1] # Calculate new center y for rotated polygon
    rotated = cv2.warpAffine(mask, M_rot, (new_w, new_h), flags=cv2.INTER_NEAREST) # Apply rotation transformation

    # Safety: if PCA gave us the orthogonal orientation, flip 90 so the object is wider than tall
    if rotated.shape[0] > rotated.shape[1]:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)

    # Long-axis length: Count of columns that contain at least one mask pixel.
    # For each column, check whether ANY pixel in that column is non-zero.
    # Then, count how many columns are True (i.e. contain at least one mask pixel).
    column_has_mask = np.any(rotated, axis=0)
    length_px = int(np.count_nonzero(column_has_mask))
    length_mm = length_px / pixels_per_mm

    # --- 4. Disk method: walk across each column, treat it as one disk, and calculate its volume
    dx_mm = 1.0 / pixels_per_mm   # Each column is 1 pixel wide, so dx is just 1px converted to mm
    disk_volumes = [] # Array to store volume of each disk
    num_columns = rotated.shape[1]

    for col_idx in range(num_columns):
        # Diameter of the object at this slice = number of non-zero pixels in the column
        diameter_px = np.count_nonzero(rotated[:, col_idx])
        if diameter_px == 0:
            continue  # Empty column (outside the object) - contributes no volume

        # Convert pixel measurements to millimeters
        diameter_mm = diameter_px / pixels_per_mm
        radius_mm = diameter_mm / 2.0

        # Volume of this disk: V = pi * r^2 * dx
        disk_volume_mm3 = math.pi * (radius_mm ** 2) * dx_mm
        disk_volumes.append(disk_volume_mm3)

    # --- 5. Calculate total volume by summing all disks
    total_vol_mm3 = sum(disk_volumes)

    return length_mm, total_vol_mm3 / 1000.0  # long-axis length in mm,  mm^3 -> mL


def find_length_and_volume(classname, polygon, pixels_per_mm):
    """Dispatch to the right estimator based on the class's shape category.
    Returns (length_mm, volume_ml).
    """
    if classname in SPHERICAL_CLASSES:
        length_mm, vol_ml = find_spherical_length_and_volume(polygon, pixels_per_mm)
        # Apply the per-class Finagling Factor to correct for the fact that real vegetables aren't perfect spheres.
        vol_ml *= FINAGLING_FACTORS.get(classname, 1.0)
        return length_mm, vol_ml
    if classname in OBLONG_CLASSES:
        return find_oblong_length_and_volume(polygon, pixels_per_mm)
    return 0.0, 0.0  # Unknown / unsupported shape category


# =========================================================================================
# Drawing helpers
# =========================================================================================

def draw_mask(frame, overlay, polygon, color, label):
    """Draw a segmentation mask, outline, and centered label — same style as yolo_segment.py."""
    cv2.fillPoly(overlay, [polygon], color)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

    # Label centered on the mask's bounding-box center
    cx = int((polygon[:, 0].min() + polygon[:, 0].max()) / 2)
    cy = int((polygon[:, 1].min() + polygon[:, 1].max()) / 2)

    font_scale = 0.4
    font_thickness = 1
    (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = cx - lw // 2
    text_y = cy + lh // 2
    cv2.rectangle(frame,
                  (text_x - 2, text_y - lh - 2),
                  (text_x + lw + 2, text_y + baseline),
                  color, cv2.FILLED)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)


def draw_panel(canvas, panel_x0, panel_w, panel_h, tallies, fps):
    """Paint the info panel on the right side of the canvas."""
    # Background
    cv2.rectangle(canvas, (panel_x0, 0), (panel_x0 + panel_w, panel_h), (30, 30, 30), cv2.FILLED)

    # Title
    cv2.putText(canvas, 'VeggieVision', (panel_x0 + 10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    # FPS readout
    cv2.putText(canvas, f'FPS: {fps:0.2f}', (panel_x0 + 10, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Divider line
    cv2.line(canvas, (panel_x0 + 5, 84), (panel_x0 + panel_w - 5, 84), (80, 80, 80), 1)

    if not tallies:
        cv2.putText(canvas, 'No vegetables detected', (panel_x0 + 10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return

    y = 114
    for name, info in tallies.items():
        count    = info['count']
        vol_ml   = info['volume_ml']
        weight_g = info['weight_g']
        vol_cups   = vol_ml / ML_PER_CUP
        weight_lbs = weight_g / G_PER_LB

        display_name = name.replace('_', ' ')
        plural = 's' if count > 1 else ''

        line1 = f'{count} {display_name}{plural}'
        line2 = f'{vol_ml:.0f} mL ({vol_cups:.1f} cups)'
        line3 = f'{weight_g:.0f} g ({weight_lbs:.1f} lbs)'

        cv2.putText(canvas, line1, (panel_x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(canvas, line2, (panel_x0 + 10, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        cv2.putText(canvas, line3, (panel_x0 + 10, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        y += 78
        if y > panel_h - 72:  # Out of vertical room
            break


# =========================================================================================
# Main loop
# =========================================================================================

def main():
    # Load the model
    model = YOLO(MODEL_PATH, task='segment')
    labels = model.names

    # Open the USB camera
    if TEST_VIDEO_NAME is not None:
        cap = cv2.VideoCapture(TEST_VIDEO_NAME)
    else:
        cap = cv2.VideoCapture(cv2.CAP_DSHOW + CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f'ERROR: Could not open USB camera at index {CAMERA_INDEX}.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-allocate the composited canvas: camera feed on the left + info panel on the right
    canvas = np.zeros((actual_h, actual_w + PANEL_W, 3), dtype=np.uint8)

    # FPS tracking
    avg_fps = 0.0
    fps_buffer = []
    fps_avg_len = 200

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret or frame is None:
            print('Unable to read frame from the camera. Exiting.')
            break

        # Run segmentation inference. retina_masks=True keeps masks in the input resolution,
        # which matters for accurate volume estimates.
        results = model(frame, verbose=False, retina_masks=True)
        detections = results[0].boxes
        masks = results[0].masks

        # Accumulate per-class tallies this frame
        tallies = {}  # { classname: {'count': int, 'volume_ml': float, 'weight_g': float} }

        overlay = frame.copy()

        if masks is not None:
            mask_polygons = masks.xy

            for i in range(len(detections)):
                conf = detections[i].conf.item()
                if conf <= CONF_THRESH:
                    continue

                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                polygon = mask_polygons[i]
                if polygon is None or len(polygon) == 0:
                    continue
                polygon = polygon.astype(np.int32)

                color = MASK_COLORS[classidx % 10]

                # Volume, length & weight
                length_mm, vol_ml = find_length_and_volume(classname, polygon, PIXELS_PER_MM)
                density = DENSITIES.get(classname, 1.0)
                weight_g = vol_ml * density

                # Accumulate tally
                if classname not in tallies:
                    tallies[classname] = {'count': 0, 'volume_ml': 0.0, 'weight_g': 0.0}
                tallies[classname]['count']     += 1
                tallies[classname]['volume_ml'] += vol_ml
                tallies[classname]['weight_g']  += weight_g

                # Draw mask + label (label shows per-item length: diameter for spherical,
                # long-axis length for oblong).
                label = f'{classname}: {length_mm:.0f} mm'
                draw_mask(frame, overlay, polygon, color, label)

        # Blend overlay into the frame for the transparent mask effect
        frame = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)

        # Composite onto the canvas: camera on the left, panel on the right
        canvas[:, :actual_w] = frame
        draw_panel(canvas, actual_w, PANEL_W, actual_h, tallies, avg_fps)

        cv2.imshow('VeggieVision', canvas)

        # Keyboard controls (same as yolo_segment.py)
        key = cv2.waitKey(100)
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey()  # Pause until any key

        # FPS bookkeeping
        frame_fps = 1.0 / (time.perf_counter() - t_start)
        if len(fps_buffer) >= fps_avg_len:
            fps_buffer.pop(0)
        fps_buffer.append(frame_fps)
        avg_fps = float(np.mean(fps_buffer))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
