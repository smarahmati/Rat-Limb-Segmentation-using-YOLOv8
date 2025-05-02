**Rat Limb Segmentation using YOLOv8**

This repository contains code for segmenting rat limbs (specifically the
right thigh, right shank, and right paw) from video frames using
[YOLOv8](https://github.com/ultralytics/ultralytics). The workflow
includes:

1.  **Manual segmentation** of selected video frames to create
    ground-truth training data.

2.  **Training** a YOLOv8 segmentation model (using PyTorch and CUDA for
    GPU acceleration).

3.  **Inference** on new videos with the trained model to generate
    segmentation results.

------------------------------------------------------------------------

**Contents**

1.  [Overview](#overview)

2.  [Environment & Dependencies](#environment--dependencies)

3.  [Scripts](#scripts)

    -   [A_Segmentation.py](#a_segmentationpy)

    -   [B_YOLO_training_GPU.py](#b_yolo_training_gpypy)

    -   [C_Apply_YOLO.py](#c_apply_yolopy)

4.  [Usage Steps](#usage-steps)

    1.  [Manual Segmentation](#1-manual-segmentation)

    2.  [Training the YOLOv8 Model](#2-training-the-yolov8-model)

    3.  [Applying the Trained Model](#3-applying-the-trained-model)

5.  [Notes on Technical Details](#notes-on-technical-details)

6.  [Acknowledgments](#acknowledgments)

------------------------------------------------------------------------

**Overview**

Our goal is to segment three parts of the rat's right limb:

-   **Right Thigh**

-   **Right Shank**

-   **Right Paw**

We use YOLOv8\'s segmentation capabilities from the [Ultralytics
package](https://pypi.org/project/ultralytics). The process is:

1.  **Collect frames** from a video and manually draw closed polygons to
    label each part of the limb.

2.  **Convert** these labeled polygons into a YOLOv8-compatible dataset
    (images & labels).

3.  **Train** a YOLOv8 segmentation model.

4.  **Run** inference on new videos to generate segmented outputs and
    optionally record the results in an Excel file.

------------------------------------------------------------------------

**Environment & Dependencies**

-   **Python 3.8+** is recommended.

-   **PyTorch** with CUDA support (if you want GPU acceleration)

-   [**Ultralytics (YOLOv8)**](https://pypi.org/project/ultralytics)

bash

pip install ultralytics

-   **OpenCV** for image & video operations

-   **Numpy, Matplotlib, Scipy, Openpyxl**

-   **Optional:** Jupyter or other IDE for easier step-by-step usage.

You can install the primary requirements with:

bash

pip install opencv-python numpy matplotlib scipy openpyxl

For GPU usage, ensure you have a compatible PyTorch install with CUDA:

bash

pip install torch torchvision torchaudio \--extra-index-url
https://download.pytorch.org/whl/cu118

*(Adjust the cu118 or cu11x part depending on your CUDA version.)*

**Scripts**

**A_Segmentation.py**

-   **Goal:** Manually annotate frames from a video with polygons
    representing each segment (thigh, shank, paw).

-   **Key libraries:**

    -   OpenCV for reading frames from the video.

    -   matplotlib for interactive polygon drawing.

    -   json & csv (with openpyxl to export).

-   **Workflow:**

    -   Sampling frames from your specified videos.

    -   Interactive polygon drawing (left clicks) with matplotlib's
        PolygonSelector.

    -   Realtime smoothing & region filling.

    -   Storing segmentation data into .json and .xlsx files.

Figure 1. Interactive polygon annotation for the rat's right thigh,
shank, and paw

**B_YOLO_training_GPU.py**

-   **Goal:** Convert manual annotations into a YOLOv8 dataset and train
    a segmentation model.

-   **Key libraries:**

    -   ultralytics.YOLO

    -   PyTorch for GPU checks.

-   **Workflow:**

    -   Reads the manual_segmentations.json (output from
        A_Segmentation.py).

    -   Creates a YOLOv8 dataset structure (images/ and labels/) with
        normalized polygon coordinates.

    -   Writes a data.yaml file for YOLOv8 specifying dataset paths &
        class names.

    -   Either starts from a base YOLOv8 segmentation model (e.g.,
        yolov8s-seg.pt) or continues from a previous training
        checkpoint.

    -   Trains the model for a specified number of epochs, saving the
        best model (best.pt) and also a final .pt checkpoint.

**C_Apply_YOLO.py**

-   **Goal:** Run inference on new videos using the trained YOLOv8 model
    to obtain segmentations in each frame.

-   **Key libraries:**

    -   OpenCV for reading and writing videos.

    -   ultralytics for loading the trained YOLO model.

    -   openpyxl for writing results to Excel.

-   **Workflow:**

    -   Loads the trained YOLO model (e.g., trained_model.pt).

    -   Iterates over frames of the input video.

    -   Predicts segmentation masks for each class (thigh, shank, paw).

    -   **Optional** morphological smoothing (closing, opening) to
        refine mask edges.

    -   Overlays the segmentation on the output video with custom
        color-coded polygons.

    -   Records the boundary and interior pixel coordinates in an Excel
        file for further analysis (frame-by-frame).

Figure 2. Comparison of the original (top) and segmented (bottom) rat
video frames.

------------------------------------------------------------------------

**Usage Steps**

Below is a stepwise outline for using these scripts. Adjust paths and
parameters as needed.

**1. Manual Segmentation**

1.  **Open A_Segmentation.py.**

    -   Update the user parameters at the top:

        -   VIDEO_PATHS with your path(s) to rat videos.

        -   NUM_FRAMES_TO_SAMPLE to control how many frames are sampled
            from each video.

        -   SEGMENTATION_NAMES which should match the segments:
            \[\"right_thigh\",\"right_shank\",\"right_paw\"\].

        -   OUTPUT_JSON_PATH (default: manual_segmentations.json).

        -   OUTPUT_CSV_PATH (default: manual_segmentations.csv / .xlsx).

2.  **Run A_Segmentation.py.**

    -   A Matplotlib window will appear for each sampled frame in
        sequence.

    -   Use **left-click** to place points forming a polygon.

    -   Press **z** to undo the last point.

    -   If the last point is near the first, it auto-closes the polygon
        and proceeds.

    -   After finishing a polygon, press **space** to confirm.

    -   Repeat for each segment (thigh, shank, paw).

    -   The script periodically **auto-saves** your annotations into
        JSON & XLSX.

**2. Training the YOLOv8 Model**

1.  **Open B_YOLO_training_GPU.py.**

    -   Confirm JSON_FILE points to the manual_segmentations.json you
        created.

    -   Update VIDEO_PATHS to match any videos used in annotation (the
        script re-loads frames to build the dataset).

    -   Set ADDITIONAL_EPOCHS (e.g., 1000).

    -   BASE_MODEL can be yolov8s-seg.pt or another YOLO segmentation
        variant.

2.  **Run B_YOLO_training_GPU.py.**

    -   The script:

        -   Loads your annotation JSON.

        -   Creates a yolo_dataset/ folder with images/ + labels/.

        -   Writes data.yaml with paths and class names.

        -   Trains YOLOv8 (GPU if available) for the specified epochs.

    -   Check the newly created seg_training/ folder (or seg_training2/,
        etc.) for logs and weights.

    -   **After training**, it copies best.pt to trained_model.pt
        (default name), which you can use for inference.

**3. Applying the Trained Model**

1.  **Open C_Apply_YOLO.py.**

    -   Set MODEL_PATH to the path of your trained model (e.g.,
        trained_model.pt).

    -   Set INPUT_VIDEO_PATH to the new video you want to segment.

    -   OUTPUT_VIDEO_PATH is where the segmented output is stored.

    -   EXCEL_FILE_PATH is where the script will write per-frame
        segmentation data (optional for analysis).

2.  **Run C_Apply_YOLO.py.**

    -   The script processes each frame of the input video:

        -   Generates segmentation masks for each limb class.

        -   Optionally applies morphological smoothing.

        -   Draws polygons onto the output video with color-coded
            overlays.

        -   Saves boundary & interior points to the Excel file.

    -   Check the final segmentation results in segmented_output.mp4 (or
        your chosen path).

------------------------------------------------------------------------

**Notes on Technical Details**

-   **YOLOv8 Segmentation**\
    We rely on the ultralytics package, which extends YOLOv8 to segment
    objects by generating class-specific masks.

-   **PyTorch & CUDA**

    -   In B_YOLO_training_GPU.py, we detect torch.cuda.is_available()
        to decide whether training runs on GPU or CPU.

    -   Training is significantly faster on GPU if you have a supported
        NVIDIA device.

-   **Morphological Operations**\
    In C_Apply_YOLO.py, we apply cv2.morphologyEx with a closing and
    opening operation to create smoother masks. You can tweak the
    KERNEL_SIZE if you want more or less smoothing.

------------------------------------------------------------------------

**Acknowledgments**

-   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for
    the core segmentation.

-   [PyTorch](https://pytorch.org/) for deep learning framework and CUDA
    support.

-   [OpenCV](https://opencv.org/) and
    [Matplotlib](https://matplotlib.org/) for image/video handling and
    user interaction.

Feel free to open issues or pull requests if you encounter problems or
want to contribute improvements!

------------------------------------------------------------------------

**Contact**

For questions or comments, please reach out to:

-   **Seyed Mohammadali Rahmati**: smarahmati@gmail.com
