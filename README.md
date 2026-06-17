# Fish-Tracking-Classification

## Overview

This pipeline automates fish detection and classification from underwater video. It combines YOLOFish for fish detection, DeepSORT-based tracking to identify unique fish across frames, and FishNet for taxonomic family classification.

The goal is to avoid counting the same fish multiple times while generating family-level classifications and confidence scores for each individual fish observed in a video.

---

## Step 1: Fish Detection with YOLOFish

YOLOFish is used to detect fish and generate bounding boxes for each frame.

### Input

* Underwater video (.mp4)

### Command

```bash
./darknet detector demo data/obj.data cfg/yolov4.cfg backup/merge_yolov4.weights OneFishTest.mp4 -dont_show -ext_output -out_filename OneFishTest_Output.mp4 -save_labels 2>&1 | tee OneFish_Output.txt
```

### Output

* Video containing fish bounding boxes
* Text file containing bounding box coordinates

Bounding box coordinates are generated approximately every 0.1 frames. For faster processing, the frame rate can be reduced to 0.5 fps.

---

## Step 2: Fish Tracking and Family Classification

`Tracking_Infernce.py` tracks fish across frames and performs family classification.

### Tracking

The script:

* Tracks fish across frames using bounding box coordinates
* Prevents the same fish from being saved multiple times
* Measures how long each fish remains visible on screen
* Saves cropped fish images into the `fish_outputs` directory

### Classification

FishNet is used to classify each cropped fish image.

The script outputs:

* Most likely family prediction
* Confidence score
* Second most likely family prediction
* Second confidence score

Results are saved to:

```text
fish_tracking_classification_results.csv
```

### Run

```bash
python Tracking_Infernce.py
```

Run from the FishNet directory.

---

## Output Files

### fish_outputs/

Contains cropped images of tracked fish.

### fish_tracking_classification_results.csv

Contains:

* Fish ID
* Tracking ID
* Predicted family
* Confidence score
* Secondary family prediction
* Secondary confidence score
* Bounding box information
* Frame tracking information

---

## Previous Version

The original workflow used separate scripts:

* `fish_tracker.py`
* `infer.py`

Results were saved to:

```text
inference_results.csv
```

Example command:

```bash
infer.py --image_path fish_outputs/fish_1.jpg --model_path weights/beit_base_patch16_224_Family.pt
```

---

## Dependencies

* YOLOFish
* Darknet
* FishNet
* DeepSORT
* PyTorch
* OpenCV
* Pandas

## Author

Alisa Vendrow

