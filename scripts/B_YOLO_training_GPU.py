import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import cv2
import shutil
import csv

import torch  # We'll use torch to check if GPU is available

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing ultralytics (YOLO v8): {e}")
    print("Run 'pip install ultralytics' first.")
    exit()

##############################################################################
# USER PARAMETERS
##############################################################################

JSON_FILE = "manual_segmentations.json"

# All actual video file paths go here
VIDEO_PATHS = [
    r"C:\Users\smara\miniconda3\envs\yolo_env\MyScripts\Segmentation-YOLO\Cam1_2024-05-01_12-44-08.mp4",
    # Add more videos if needed...
]

OUTPUT_DATASET_DIR = "yolo_dataset"
DATA_YAML_PATH = "data.yaml"
MODEL_SAVE_PATH = "trained_model.pt"
BASE_MODEL = "yolov8s-seg.pt"

ADDITIONAL_EPOCHS = 1000    # Number of epochs to train in the new run
IMGSZ = 640               # Image size used for training
SAVE_PERIOD = 10          # YOLO will save a checkpoint every epoch
SAVE_CHECKPOINT_START = 20  # keep only epoch >= 20
SAVE_CHECKPOINT_EVERY = 10  # keep only multiples of 10 (20,30,40,...)

# Classes => numeric IDs
SEGMENT_NAME_TO_ID = {
    "right_thigh": 0,
    "right_shank": 1,
    "right_paw":   2
}

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def override_json_paths(all_data, user_video_paths):
    """
    For each entry in 'all_data', replace the 'video_path' with a matching user-defined path
    if the base filename matches. E.g., if JSON has "C:/old/path/Cam1_2024-05-01.mp4"
    and user_video_paths has "D:/new/path/Cam1_2024-05-01.mp4", override with the latter.
    """
    name_to_fullpath = {}
    for vp in user_video_paths:
        base_name = os.path.basename(vp)
        name_to_fullpath[base_name] = vp

    for vidinfo in all_data:
        orig_path = vidinfo.get("video_path", "")
        base_name = os.path.basename(orig_path)
        if base_name in name_to_fullpath:
            vidinfo["video_path"] = name_to_fullpath[base_name]
        else:
            print(f"Warning: no override found for {base_name} => remains {orig_path}")
    return all_data

def load_json(json_file):
    if not os.path.isfile(json_file):
        print(f"Error: JSON file not found => {json_file}")
        return []
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def build_frame_map(all_data):
    frame_map = {}
    for vidinfo in all_data:
        vp   = vidinfo["video_path"]
        segs = vidinfo["segmentation_data"]
        for sd in segs:
            fidx  = sd.get("sample_frame_index", None)
            sname = sd.get("seg_name", None)
            poly  = sd.get("polygon", [])
            if fidx is None or sname is None or not poly:
                continue
            frame_map.setdefault((vp, fidx), []).append((sname, poly))
    return frame_map

def create_yolo_dataset(frame_map):
    images_dir = os.path.join(OUTPUT_DATASET_DIR, "images")
    labels_dir = os.path.join(OUTPUT_DATASET_DIR, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for (vp, fidx), seg_list in frame_map.items():
        if not os.path.isfile(vp):
            print(f"Warning: video '{vp}' not found => skip.")
            continue
        cap = cv2.VideoCapture(vp)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fidx >= total_f:
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            continue

        h, w, _ = frame.shape
        out_img_name = f"{os.path.basename(vp).replace('.mp4','')}_frame{fidx}.jpg"
        img_path = os.path.join(images_dir, out_img_name)
        cv2.imwrite(img_path, frame)

        lines = []
        for (sname, poly) in seg_list:
            cid = SEGMENT_NAME_TO_ID.get(sname, 0)
            norm_coords = []
            for (xx, yy) in poly:
                xN = xx / float(w)
                yN = yy / float(h)
                norm_coords.append(xN)
                norm_coords.append(yN)
            line_str = str(cid) + " " + " ".join(f"{v:.6f}" for v in norm_coords)
            lines.append(line_str)

        txt_name = out_img_name.replace(".jpg", ".txt")
        txt_path = os.path.join(labels_dir, txt_name)
        with open(txt_path, 'w') as f:
            for ln in lines:
                f.write(ln + "\n")

def write_data_yaml(classes):
    abs_dataset_dir = os.path.abspath(OUTPUT_DATASET_DIR)
    with open(DATA_YAML_PATH, 'w') as f:
        f.write(f"train: {abs_dataset_dir}/images\n")
        f.write(f"val: {abs_dataset_dir}/images\n")
        f.write(f"test: {abs_dataset_dir}/images\n\n")
        f.write("names:\n")
        for i, cname in enumerate(classes):
            f.write(f"  {i}: {cname}\n")

def have_previous_model():
    """
    Returns True if 'seg_training/weights/last.pt' exists, indicating
    we have a previously trained model to start from.
    """
    last_pt_path = os.path.join("seg_training", "weights", "last.pt")
    return os.path.isfile(last_pt_path)

def get_next_run_folder():
    """
    Returns the next run folder name in the sequence:
      seg_training, seg_training2, seg_training3, ...
    It checks each possibility until it finds a folder name
    that does not exist on disk.
    """
    base = "seg_training"
    i = 1
    while True:
        if i == 1:
            name = base          # seg_training
        else:
            name = f"{base}{i}"  # seg_training2, seg_training3, ...
        if not os.path.exists(name):
            return name
        i += 1

def post_process_checkpoints(start_epoch, step):
    """
    After training, keep only epoch_X >= start_epoch in multiples of 'step' (20,30,40...).
    Remove the rest. This checks *all* seg_training folders that exist.
    """
    for folder in os.listdir("."):
        if folder.startswith("seg_training"):  # e.g. seg_training, seg_training2, ...
            weights_dir = os.path.join(folder, "weights")
            if not os.path.isdir(weights_dir):
                continue
            for fname in os.listdir(weights_dir):
                if not fname.startswith("epoch_"):
                    continue
                parts = fname.split("_")
                if len(parts) < 2:
                    continue
                epoch_str = parts[1].replace(".pt","")
                try:
                    epoch_num = int(epoch_str)
                except:
                    continue
                # Remove if epoch_num < start_epoch or not a multiple of step
                if (epoch_num < start_epoch) or (epoch_num % step != 0):
                    path = os.path.join(weights_dir, fname)
                    os.remove(path)

##############################################################################
# TRAIN (GPU if available)
##############################################################################

def train_yolo_seg():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    # Decide if we have an existing model to continue from
    prev_exists = have_previous_model()

    # Generate a new run folder name each time
    run_folder = get_next_run_folder()
    print(f"Chosen run folder name => {run_folder}")

    if prev_exists:
        print(f"Found existing model => starting a *new* run from 'last.pt' for {ADDITIONAL_EPOCHS} epochs.")
        model = YOLO(os.path.join("seg_training", "weights", "last.pt"))
        # Train as a fresh run, not resume
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=ADDITIONAL_EPOCHS,
            imgsz=IMGSZ,  # <---- Use IMGSZ here
            project=".",
            name=run_folder,
            save_period=SAVE_PERIOD,
            device=device_str,
            resume=False
        )
    else:
        print(f"No existing training => starting from base model ({BASE_MODEL}) "
              f"for {ADDITIONAL_EPOCHS} epochs.")
        model = YOLO(BASE_MODEL)
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=ADDITIONAL_EPOCHS,
            imgsz=IMGSZ,  # <---- Use IMGSZ here
            project=".",
            name=run_folder,
            save=True,
            save_period=SAVE_PERIOD,
            device=device_str
        )

    # Copy best.pt => MODEL_SAVE_PATH from the newly used run folder
    best_weights = os.path.join(run_folder, "weights", "best.pt")
    if os.path.isfile(best_weights):
        shutil.copy(best_weights, MODEL_SAVE_PATH)
        print(f"Saved final YOLO weights => {MODEL_SAVE_PATH}")
    else:
        print("Warning: best weights not found => check training folder.")

    # Prune checkpoints if desired
    post_process_checkpoints(SAVE_CHECKPOINT_START, SAVE_CHECKPOINT_EVERY)

##############################################################################
# MAIN
##############################################################################

def main():
    # 1) Load JSON
    if not os.path.isfile(JSON_FILE):
        print(f"Error: '{JSON_FILE}' not found => can't proceed.")
        return
    with open(JSON_FILE, 'r') as f:
        all_data = json.load(f)
    if not all_data:
        print("JSON empty => no polygons => exit.")
        return
    print(f"Loaded => {JSON_FILE}")

    # 2) Override video paths from user-provided VIDEO_PATHS
    all_data = override_json_paths(all_data, VIDEO_PATHS)

    # 3) Build the dataset => yolo_dataset
    frame_map = build_frame_map(all_data)
    create_yolo_dataset(frame_map)
    print(f"Created YOLO dataset in '{OUTPUT_DATASET_DIR}'")

    # 4) Build data.yaml
    max_id = max(SEGMENT_NAME_TO_ID.values())
    class_count = max_id + 1
    names = [None]*class_count
    for k, v in SEGMENT_NAME_TO_ID.items():
        names[v] = k if k else f"class{v}"
    write_data_yaml(names)
    print(f"Wrote '{DATA_YAML_PATH}' with absolute paths")

    # 5) Train a new run (from scratch or from last.pt)
    train_yolo_seg()
    print("Done. Checkpoints are pruned after training. Final model =>", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
