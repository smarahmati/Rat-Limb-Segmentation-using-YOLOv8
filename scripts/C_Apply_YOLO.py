import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from openpyxl import Workbook

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics (YOLOv8) not installed. Run 'pip install ultralytics' first.")
    exit()


##############################################################################
# USER PARAMETERS
##############################################################################

MODEL_PATH        = r"C:\Users\smara\miniconda3\envs\yolo_env\MyScripts\Segmentation-YOLO\trained_model.pt"
INPUT_VIDEO_PATH  = r"C:\Users\smara\miniconda3\envs\yolo_env\MyScripts\Segmentation-YOLO\Cam1_2024-05-01_12-44-08.mp4"
OUTPUT_VIDEO_PATH = r"C:\Users\smara\miniconda3\envs\yolo_env\MyScripts\Segmentation-YOLO\segmented_output.mp4"
EXCEL_FILE_PATH   = r"C:\Users\smara\miniconda3\envs\yolo_env\MyScripts\Segmentation-YOLO\results.xlsx"

CONF_THRESHOLD = 0.5       # YOLO detection confidence threshold
IMGSZ         = 640        # YOLO inference size

LINE_WIDTH    = 2          # thickness of boundary lines
REGION_ALPHA  = 0.3        # transparency for fill
STRIDE        = 5          # interior downsampling: keep every 5th pixel
NUM_POINTS    = 200        # sample this many boundary points on each contour

# Morphological kernel for smoothing
KERNEL_SIZE = 10  # bigger => more smoothing
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# "both" => draw raw YOLO polygons + morphological-smoothed polygons
# "only_curve" => only morphological polygons
# "only_lines" => only raw YOLO polygons
SHOW_MODE = "only_curve"

# BGR colors for classes 0..N
# The classes numer is related to
# SEGMENT_COLORS = {
#     "right_thigh": "r",
#     "right_shank": "g",
#     "right_paw":   "b",
# }
# from A_Segmentation

CLASS_COLORS = [
    (0,   0, 255),   # Red
    (0, 255,   0),   # Green
    (255, 0,   0),   # Blue
    (255, 255, 0),   # Cyan
]


##############################################################################
# HELPER FUNCTIONS
##############################################################################

def draw_polygon(frame, polygon_xy, color, thickness=2):
    """
    Draw Nx2 polygon on 'frame' in BGR color.
    """
    pts = np.array(polygon_xy, dtype=np.int32).reshape((-1,1,2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

def fill_polygon_transparent(frame, polygon_xy, color, alpha=0.3):
    """
    Fill Nx2 polygon on 'frame' with 'color' at 'alpha' transparency.
    """
    overlay = frame.copy()
    pts = np.array(polygon_xy, dtype=np.int32).reshape((-1,1,2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

def sample_contour_points(contour_xy, num_points=100):
    """
    Sample 'num_points' evenly along the contour perimeter.
    contour_xy: Nx2 array.
    Returns a list of (x,y) with length up to 'num_points'.
    """
    if len(contour_xy) < 2:
        return contour_xy.tolist()

    # Convert to float32
    arr = contour_xy.astype(np.float32)
    # Ensure closed
    if not np.all(arr[0] == arr[-1]):
        arr = np.vstack([arr, arr[0]])

    # Compute cumulative arc length
    distances = [0.0]
    for i in range(1, len(arr)):
        seg_len = np.hypot(arr[i,0] - arr[i-1,0], arr[i,1] - arr[i-1,1])
        distances.append(distances[-1] + seg_len)
    total_len = distances[-1]
    if total_len < 1e-5:
        return contour_xy.tolist()  # degenerate shape

    # Equally spaced distances
    step = total_len / float(num_points)
    sampled = []
    idx = 0
    for target_dist in np.arange(0, total_len, step):
        # Move along distances[] until we pass target_dist
        while idx < len(distances)-1 and distances[idx+1] < target_dist:
            idx += 1
        # Interpolate between arr[idx] and arr[idx+1]
        if idx == len(distances)-1:
            sampled.append(arr[-1])
        else:
            # fraction
            d1 = distances[idx]
            d2 = distances[idx+1]
            ratio = 0.0 if (d2 - d1) < 1e-5 else (target_dist - d1)/(d2 - d1)
            xA, yA = arr[idx]
            xB, yB = arr[idx+1]
            x = xA + ratio*(xB - xA)
            y = yA + ratio*(yB - yA)
            sampled.append([x,y])

    # ensure closed
    if len(sampled)>1 and not np.allclose(sampled[0], sampled[-1], atol=1e-5):
        sampled.append(sampled[0])
    return np.array(sampled, dtype=np.float32)

def get_interior_pixels_downsample(width, height, polygon_xy, stride=10):
    """
    Return interior pixel coords for Nx2 polygon 'polygon_xy', skipping by 'stride'.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon_xy, dtype=np.int32).reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 1)

    ys, xs = np.where(mask == 1)
    coords = []
    for (x, y) in zip(xs, ys):
        if x % stride == 0 and y % stride == 0:
            coords.append((x, y))
    return coords

def mask_to_contours(mask_bin):
    """
    Convert a binary mask (0/255) to a list of Nx2 contours.
    """
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if len(cnt) >= 3:
            out.append(cnt.reshape(-1,2))  # Nx2
    return out


##############################################################################
# MAIN
##############################################################################

def main():
    # 1) Load YOLO model
    model = YOLO(MODEL_PATH)
    print(f"Using model => {MODEL_PATH}")

    # 2) Open video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3) Output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    print(f"Writing => {OUTPUT_VIDEO_PATH} ({width}x{height}, {fps} fps)")

    # 4) Excel workbook => 1 sheet per frame
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End or read error.")
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ,
                                retina_masks=True, verbose=False)
        sheet_name = f"frame_{frame_idx}"
        ws = wb.create_sheet(title=sheet_name)
        ws.append(["video_path","frame_index","class_id","confidence","x","y","type"])

        if len(results) > 0:
            res = results[0]
            boxes = res.boxes
            masks = res.masks
            rows_buffer = []
            if boxes is not None and masks is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf   = float(boxes.conf[i].item())
                    if conf < CONF_THRESHOLD:
                        continue
                    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

                    # RAW YOLO polygons in original coords
                    raw_polys = masks.xy[i]
                    if not isinstance(raw_polys, list):
                        raw_polys = [raw_polys]

                    # SHOW raw polygons
                    if SHOW_MODE in ["both","only_lines"]:
                        for sub_poly in raw_polys:
                            draw_polygon(frame, sub_poly, color, LINE_WIDTH)

                    # Morphological smoothing
                    # => get the 2D mask in original size
                    mask_2d = masks.data[i].cpu().numpy()  # shape (height,width)
                    mask_bin = (mask_2d*255).astype(np.uint8)

                    # morphological ops
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, KERNEL)
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, KERNEL)

                    # find final smoothed contours
                    smoothed_contours = mask_to_contours(mask_bin)

                    # draw the "smoothed" polygons
                    if SHOW_MODE in ["both","only_curve"]:
                        for contour_xy in smoothed_contours:
                            draw_polygon(frame, contour_xy, color, LINE_WIDTH)

                    # fill
                    for contour_xy in smoothed_contours:
                        fill_polygon_transparent(frame, contour_xy, color, alpha=REGION_ALPHA)

                    # Now record boundary & interior in Excel
                    for contour_xy in smoothed_contours:
                        # 1) Sample boundary => 'NUM_POINTS'
                        boundary_pts = sample_contour_points(contour_xy, NUM_POINTS)
                        for (bx, by) in boundary_pts:
                            rows_buffer.append([
                                INPUT_VIDEO_PATH,
                                frame_idx,
                                cls_id,
                                round(conf,3),
                                int(round(bx)),
                                int(round(by)),
                                "poly"
                            ])
                        # 2) Interior => skip most => 'STRIDE'
                        interior = get_interior_pixels_downsample(width, height, boundary_pts, stride=STRIDE)
                        for (ix, iy) in interior:
                            rows_buffer.append([
                                INPUT_VIDEO_PATH,
                                frame_idx,
                                cls_id,
                                round(conf,3),
                                ix,
                                iy,
                                "pixel"
                            ])

            # after all detections in this frame
            for row in rows_buffer:
                ws.append(row)

        out_video.write(frame)
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{total_frames}")

    cap.release()
    out_video.release()
    wb.save(EXCEL_FILE_PATH)
    print(f"\nDone. Excel => {EXCEL_FILE_PATH}, Video => {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
