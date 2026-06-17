import os
import cv2
import re
import csv
import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============================================================================
# CONFIGURATION
# ============================================================================
device = 'cpu'

# Model configuration
final_layers = {
    'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048, 'resnet152': 2048,
    'vgg16': 4096, 'vit_base_patch16_224': 768, 'vit_small_patch16_224': 384,
    'vit_large_patch16_224': 1024, 'beit_base_patch16_224': 768
}

map_models = [
    'vit_base_patch16_224', 'vit_small_patch16_224', 'vit_large_patch16_224',
    'resnet34', 'resnet50', 'resnet101', 'resnet152', 'beit_base_patch16_224',
    'convnext_large'
]

model_id = 7  # beit
nClass = 463

# Preprocessing for inference
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================================
# MODEL CLASSES
# ============================================================================
class Classifiers(nn.Module):
    def __init__(self, backbone, finetune=False, use_two_fcs=False):
        super().__init__()
        modules = list(backbone.children())[:-1]
        self.model = nn.Sequential(*modules)
        for param in self.model.parameters():
            param.requires_grad = True if finetune else False

        if use_two_fcs:
            self.linear = nn.Sequential(
                nn.Linear(final_layers[map_models[model_id]], 512),
                nn.Dropout(0.5),
                nn.Linear(512, nClass)
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(final_layers[map_models[model_id]], nClass)
            )

    def forward(self, inputs):
        out = self.model(inputs).squeeze()
        if len(out.size()) == 3:
            out = out.mean(1)
        return self.linear(out)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_model(model_path):
    """Load the trained fish classification model"""
    model = timm.create_model(map_models[model_id], pretrained=False, num_classes=nClass)
    if not ('beit' in map_models[model_id] or 'convnext' in map_models[model_id]):
        model = Classifiers(model, finetune=False, use_two_fcs=True)

    pretrained_state_dict = torch.load(model_path, map_location=device)

    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def predict_image(image_path, model):
    """Predict fish family group from image and return top 2 predictions with confidence"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor).squeeze()
        probabilities = torch.softmax(outputs, dim=0)
        top2_confidences, top2_predicted = torch.topk(probabilities, 2)
    
    # Return top 2 predictions and their confidences
    return (top2_predicted[0].item(), top2_confidences[0].item(),
            top2_predicted[1].item(), top2_confidences[1].item())

def load_yolofish_detections(txt_file, confidence_thresh=70):
    """Parse YOLOFish detection log format"""
    detections = defaultdict(list)
    frame_idx = 0
    current_boxes = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_boxes:
                    detections[frame_idx] = current_boxes
                    frame_idx += 1
                    current_boxes = []
                continue
            
            match = re.search(r'Fish:\s*(\d+)%\s+\(left_x:\s*(\d+)\s+top_y:\s*(\d+)\s+width:\s*(\d+)\s+height:\s*(\d+)', line)
            if match:
                conf, x, y, w, h = map(int, match.groups())
                if conf >= confidence_thresh and w >= 30 and h >= 30:
                    current_boxes.append((x, y, w, h))
        
        if current_boxes:
            detections[frame_idx] = current_boxes
    
    return detections

# ============================================================================
# MAIN TRACKING AND CLASSIFICATION FUNCTION
# ============================================================================
def track_classify_and_save(video_path, bbox_txt_path, model_path, csv_mapping_file, 
                           output_folder, final_csv_path):
    """
    Main function that tracks fish, crops them, classifies them, and saves to unified CSV
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model and group-to-family mapping
    print("Loading classification model...")
    model = load_model(model_path)
    
    print("Loading group-to-family mapping...")
    meta_df = pd.read_csv(csv_mapping_file, header=None)
    group_to_family = dict(zip(meta_df[0], meta_df[3]))  # Column 0 -> Column 3
    
    # Load detections
    print("Loading detections...")
    detections = load_yolofish_detections(bbox_txt_path)
    
    # Initialize video capture and tracker
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = DeepSort(max_age=10, n_init=3)
    
    # Tracking variables
    saved_ids = set()
    frame_counts = defaultdict(int)
    frame_num = 0
    fish_counter = 1
    
    # CSV results storage
    csv_results = []
    
    print("Starting tracking and classification...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_dets = detections.get(frame_num, [])
        input_dets = [([x, y, w, h], 1.0, 'fish') for (x, y, w, h) in frame_dets]
        
        tracks = tracker.update_tracks(input_dets, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            frame_counts[track_id] += 1
            
            l, t, r, b = map(int, track.to_ltrb())
            l = max(0, min(l, width - 1))
            t = max(0, min(t, height - 1))
            r = max(0, min(r, width - 1))
            b = max(0, min(b, height - 1))
            
            # Save and classify once per fish, if visible > 5 frames
            if (track_id not in saved_ids) and (frame_counts[track_id] > 5):
                crop = frame[t:b, l:r]
                if crop.size > 0:
                    # Save cropped image
                    fish_filename = f"fish_{fish_counter}.jpg"
                    out_path = os.path.join(output_folder, fish_filename)
                    cv2.imwrite(out_path, crop)
                    
                    # Classify the fish
                    try:
                        predicted_group, confidence, predicted_group_2, confidence_2 = predict_image(out_path, model)
                        family = group_to_family.get(predicted_group, 'Unknown')
                        family_2 = group_to_family.get(predicted_group_2, 'Unknown')
                        confidence_percent = round(confidence * 100, 2)
                        confidence_2_percent = round(confidence_2 * 100, 2)
                        
                        # Add to CSV results (we'll update frame count later)
                        csv_results.append({
                            'Fish_ID': fish_counter,
                            'Track_ID': track_id,
                            'Image': fish_filename,
                            'First_Seen_Frame': frame_num,
                            'Frames_Tracked': 0,  # Will be updated after tracking completes
                            'Group': predicted_group,
                            'Family': family,
                            'Confidence_Percent': confidence_percent,
                            'Group_2': predicted_group_2,
                            'Family_2': family_2,
                            'Confidence_2_Percent': confidence_2_percent,
                            'Bbox_Left': l,
                            'Bbox_Top': t,
                            'Bbox_Right': r,
                            'Bbox_Bottom': b
                        })
                        
                        print(f"Fish {fish_counter} (Track {track_id}) -> 1st: Group {predicted_group}, Family: {family} ({confidence_percent}%) | 2nd: Group {predicted_group_2}, Family: {family_2} ({confidence_2_percent}%)")
                        
                    except Exception as e:
                        print(f"Error classifying fish {fish_counter}: {e}")
                        csv_results.append({
                            'Fish_ID': fish_counter,
                            'Track_ID': track_id,
                            'Image': fish_filename,
                            'First_Seen_Frame': frame_num,
                            'Frames_Tracked': 0,  # Will be updated after tracking completes
                            'Group': 'Error',
                            'Family': 'Error',
                            'Confidence_Percent': 0.0,
                            'Group_2': 'Error',
                            'Family_2': 'Error',
                            'Confidence_2_Percent': 0.0,
                            'Bbox_Left': l,
                            'Bbox_Top': t,
                            'Bbox_Right': r,
                            'Bbox_Bottom': b
                        })
                    
                    saved_ids.add(track_id)
                    fish_counter += 1
        
        frame_num += 1
        
        # Progress indicator
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames...")
    
    cap.release()
    
    # Update frame counts in CSV results after tracking is complete
    print("Updating final frame counts...")
    for result in csv_results:
        track_id = result['Track_ID']
        result['Frames_Tracked'] = frame_counts[track_id]
    
    # Save unified CSV results
    print(f"\nSaving results to {final_csv_path}...")
    with open(final_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Fish_ID', 'Track_ID', 'Image', 'First_Seen_Frame', 'Frames_Tracked', 
                     'Group', 'Family', 'Confidence_Percent', 'Group_2', 'Family_2', 'Confidence_2_Percent',
                     'Bbox_Left', 'Bbox_Top', 'Bbox_Right', 'Bbox_Bottom']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_results:
            writer.writerow(row)
    
    # Print summary
    print(" SUMMARY:")
    print(f"Total fish tracked and classified: {fish_counter-1}")
    print(f"Results saved to: {final_csv_path}")
    print(f"Fish images saved to: {output_folder}")
    
    print(" Fish track durations (frames alive):")
    for track_id, count in frame_counts.items():
        print(f"Track {track_id}: {count} frames")
    
    return csv_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Configuration paths
    video_path = "output2.mp4"
    bbox_txt_path = "output2.txt"
    model_path = "weights/beit_base_patch16_224_Family.pt"
    csv_mapping_file = "anns/train_full_meta_new.csv"
    output_folder = "fish_outputs"
    final_csv_path = "output2_fish_tracking_classification_results.csv"
    
    # Run the unified tracking and classification
    results = track_classify_and_save(
        video_path=video_path,
        bbox_txt_path=bbox_txt_path,
        model_path=model_path,
        csv_mapping_file=csv_mapping_file,
        output_folder=output_folder,
        final_csv_path=final_csv_path
    )
    
    print(f"Complete! Check '{final_csv_path}' for all results.")