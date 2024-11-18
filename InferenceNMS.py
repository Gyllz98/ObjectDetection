import os
import torch
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_proposals(img_name, proposals_dir):
    """
    Load proposals for a given image from the labeled_proposals.txt file.

    Args:
        img_name (str): Name of the image file (e.g., 'img-565.jpg')
        proposals_dir (str): Directory where the labeled proposals are stored.

    Returns:
        List[dict]: A list of proposals, each as a dict with bbox and label.
    """
    proposals = []
    base_name = os.path.splitext(img_name)[0]  # 'img-565'

    # Construct the proposal file path using forward slashes
    proposals_file = f"{proposals_dir}/{base_name}_labeled_proposals.txt"

    # Alternatively, if your files use hyphens:
    # proposals_file = f"{proposals_dir}/{base_name}-labeled_proposals.txt"

    if not os.path.exists(proposals_file):
        print(f"Proposals file not found: {proposals_file}")
        return proposals  # Return empty list

    with open(proposals_file, 'r') as f:
        for line in f:
            xmin, ymin, xmax, ymax, label = line.strip().split(',')
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            proposals.append({'bbox': bbox, 'label': label})

    return proposals



def run_model_on_proposals(image, proposals, model, device, score_threshold=0.5):
    """
    Run the trained model on the proposals extracted from an image,
    returning only those classified as 'pothole'.

    Args:
        image (PIL.Image): The input image.
        proposals (List[dict]): List of proposals with bounding boxes.
        model (torch.nn.Module): The trained model.
        score_threshold (float): Confidence score threshold for filtering detections.

    Returns:
        Tuple[List[List[int]], torch.Tensor]: The filtered bounding boxes and their corresponding scores.
    """
    imgs = []
    bboxes = []
    for proposal in proposals:
        bbox = proposal['bbox']
        x_min, y_min, x_max, y_max = bbox
        crop = image.crop((x_min, y_min, x_max, y_max))
        img = transform(crop)
        imgs.append(img)
        bboxes.append(bbox)
    if not imgs:
        return [], []
    imgs = torch.stack(imgs).to(device)  # Move batch to GPU
    with torch.no_grad():
        outputs = model(imgs)
        # Assuming binary classification; adjust based on your model's output
        scores = torch.sigmoid(outputs).squeeze()
    
    # Filter detections based on the score threshold
    keep_indices = scores >= score_threshold
    filtered_bboxes = [bbox for bbox, keep in zip(bboxes, keep_indices) if keep]
    filtered_scores = scores[keep_indices]
    
    return filtered_bboxes, filtered_scores


def get_detections_for_image(image_path, proposals_dir, model, device, score_threshold=0.5):
    """
    Get detections for a single image, returning only those classified as 'pothole'.

    Args:
        image_path (str): Path to the image file.
        proposals_dir (str): Directory where the labeled proposals are stored.
        model (torch.nn.Module): The trained model.
        score_threshold (float): Confidence score threshold for filtering detections.

    Returns:
        List[dict]: List of filtered detections with bounding boxes and scores.
    """
    image = Image.open(image_path).convert("RGB")
    img_name = os.path.basename(image_path)  # e.g., 'img-565.jpg'

    # Load proposals
    proposals = load_proposals(img_name, proposals_dir)

    # Run model on proposals and filter detections
    bboxes, scores = run_model_on_proposals(image, proposals, model, device, score_threshold)
    detections = []
    for bbox, score in zip(bboxes, scores):
        detections.append({'bbox': bbox, 'score': score.item(), 'image_id': img_name})

    return detections


def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.

    Args:
        detections (List[dict]): List of detections with bounding boxes and scores.
        iou_threshold (float): Intersection over Union (IoU) threshold.

    Returns:
        List[dict]: Detections after NMS.
    """
    boxes = [det['bbox'] for det in detections]
    scores = [det['score'] for det in detections]
    if not boxes:
        return []
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores)
    indices = nms(boxes, scores, iou_threshold)
    nms_detections = [detections[i] for i in indices]
    return nms_detections