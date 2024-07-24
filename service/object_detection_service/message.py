import textdistance
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from torchvision.ops import nms
import cv2
import pandas as pd
import numpy as np
import textdistance
from sklearn.cluster import DBSCAN


# OCR Detection Function
def paddle_OCR(image_path, ocr_model):
    results = ocr_model.ocr(image_path, cls=True)
    new_results_df = convert_OCR_result(results)
    return new_results_df[new_results_df['confidence'] > 0.7]
def map_ocr_to_brands(df_OCR, threshold=0.3):
    def find_best_match(text):
        best_match, best_score = None, float('inf')
        all_brands = heineken_brand_list + beer_competitor_list + soft_drink_competitor_list + mineral_water_competitor_list
        for brand in all_brands:
            distance = textdistance.levenshtein.normalized_distance(text.lower(), brand.lower())
            if distance < best_score:
                best_match, best_score = brand, distance
        return best_match if best_score <= threshold else None

    df_OCR['name'] = df_OCR['name'].apply(find_best_match)
    
    def classify_brand(name):
        if name in heineken_brand_list:
            return f"{name}"
        elif name in beer_competitor_list:
            return f"(beer competitor) {name}"
        elif name in soft_drink_competitor_list:
            return f"(soft drink competitor) {name}"
        elif name in mineral_water_competitor_list:
            return f"(mineral water competitor) {name}"
        else:
            return None
    
    df_OCR['new_name'] = df_OCR['name'].apply(classify_brand)
    df_OCR = df_OCR[df_OCR['new_name'].notna()]
    
    return df_OCR

def calculate_centroids(rectangles):
    rectangles['cx'] = (rectangles['xmin'] + rectangles['xmax']) / 2
    rectangles['cy'] = (rectangles['ymin'] + rectangles['ymax']) / 2
    return rectangles

def calculate_eps(image):
    image_height, image_width = image.shape[:2]
    eps = max(image_width / 5, image_height / 5)
    return eps

def find_groups_dbscan(rectangles, image, min_samples=4):
    rectangles = calculate_centroids(rectangles)
    centroids = rectangles[['cx', 'cy']].values
    eps = calculate_eps(image)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    rectangles['group'] = clustering.labels_

    outliers = rectangles[rectangles['group'] == -1].index
    for i, outlier in enumerate(outliers):
        rectangles.at[outlier, 'group'] = max(rectangles['group']) + 1 + i

    return rectangles

def process_ocr_data(df_OCR, image):
    df_OCR = map_ocr_to_brands(df_OCR)
    df_OCR = find_groups_dbscan(df_OCR, image)
    
    # Tính toán số dòng trong mỗi nhóm
    group_sizes = df_OCR.groupby('group').size().reset_index(name='group_size')
    
    # Gán số lượng nhóm vào dataframe ban đầu
    df_OCR = df_OCR.merge(group_sizes, on='group', how='left')
    
    # Xác định brand_class dựa trên số lượng dòng trong nhóm
    def determine_brand_class(row):
        if row['group_size'] == 1:
            return f"{row['new_name']} logo"
        elif row['group_size'] > 2:
            return f"{row['new_name']} beer carton"
        return None
    
    df_OCR['brand_class'] = df_OCR.apply(determine_brand_class, axis=1)
    df_OCR = df_OCR[df_OCR['brand_class'].notna()]

    df_OCR = df_OCR[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name', 'brand_class']]
    
    return df_OCR

def detect_person(image):
    model = YOLOv10(os.path.join(weights_dir, 'yolov10x.pt'))
    results = model(image)[0]
    boxes = results.boxes
    xyxys = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    results_df = pd.DataFrame({
        'xmin': xyxys[:, 0],
        'ymin': xyxys[:, 1],
        'xmax': xyxys[:, 2],
        'ymax': xyxys[:, 3],
        'confidence': scores,
        'name': classes
    })
    results_df = results_df[results_df['name'] == 0]
    results_df['name'] = 'person'

    return results_df

# Object Detection Function
def owlv2_detection(im):
    checkpoint = "google/owlv2-base-patch16-ensemble"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)
    text_queries = ["beer bottle", "beer can", "beer carton", "beer crate", "ice bucket", "ice box", "fridge", "signage", "billboard", "poster", "standee", "tent card", "display stand", "tabletop display", "parasol"]
    thresholds = {"beer bottle": 0.3, "beer can": 0.3, "beer carton": 0.4, "beer crate": 0.4, "ice bucket": 0.4, "ice box": 0.4, "fridge": 0.4, "signage": 0.4, "billboard": 0.3, "poster": 0.3, "standee": 0.3, "tent card": 0.4, "display stand": 0.4, "tabletop display": 0.4, "parasol": 0.4}

    inputs = processor(text=text_queries, images=im, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

    scores = torch.tensor(results["scores"])
    labels = torch.tensor(results["labels"])
    boxes = torch.tensor(results["boxes"])

    final_boxes, final_scores, final_labels = [], [], []
    unique_labels = labels.unique()
    for ul in unique_labels:
        mask = labels == ul
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        nms_indices = nms(filtered_boxes, filtered_scores, iou_threshold=0.3)
        final_boxes.append(filtered_boxes[nms_indices])
        final_scores.append(filtered_scores[nms_indices])
        final_labels.append(filtered_labels[nms_indices])

    if final_boxes:
        final_boxes = torch.cat(final_boxes)
        final_scores = torch.cat(final_scores)
        final_labels = torch.cat(final_labels)
    else:
        final_boxes = torch.tensor([])
        final_scores = torch.tensor([])
        final_labels = torch.tensor([])

    data = {"xmin": [], "ymin": [], "xmax": [], "ymax": [], "confidence": [], "name": []}
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        label_text = text_queries[label]
        if score.item() >= thresholds[label_text]:
            box = [round(i, 2) for i in box.tolist()]
            data["xmin"].append(box[0])
            data["ymin"].append(box[1])
            data["xmax"].append(box[2])
            data["ymax"].append(box[3])
            data["confidence"].append(round(score.item(), 3))
            data["name"].append(label_text)
    df_OD = pd.DataFrame(data)
    return df_OD

# Function to check if one bounding box is within another
def is_within(box1, box2, threshold=0.5):
    # return box1['xmin'] >= box2['xmin'] and box1['ymin'] >= box2['ymin'] and box1['xmax'] <= box2['xmax'] and box1['ymax'] <= box2['ymax']

    # Calculate the coordinates of the intersection rectangle
    inter_xmin = max(box1['xmin'], box2['xmin'])
    inter_ymin = max(box1['ymin'], box2['ymin'])
    inter_xmax = min(box1['xmax'], box2['xmax'])
    inter_ymax = min(box1['ymax'], box2['ymax'])

    # Calculate the area of the intersection rectangle
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # Calculate the area of box1
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])

    # Calculate the ratio of the intersection area to the area of box1
    ratio = inter_area / box1_area

    # Check if the ratio exceeds the threshold
    return ratio >= threshold

# Main function to process and create the final dataframe
def process_image(image_path, df_OCR):

    im = Image.open(image_path)
    opencv_image = np.array(im)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    df_owlv2 = owlv2_detection(im)

    # Create a list to keep track of indices to drop
    indices_to_drop = set()

    for i, box1 in df_owlv2.iterrows():
        for j, box2 in df_owlv2.iterrows():
            if i != j and is_within(box1, box2, threshold=1.0):  # Use threshold=1.0 for 100% coverage
                if (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin']) < (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin']):
                    indices_to_drop.add(i)
                else:
                    indices_to_drop.add(j)

    df_owlv2 = df_owlv2.drop(indices_to_drop)

    df_person = detect_person(im)

    df_OD = pd.concat([df_owlv2, df_person], axis=0)

    # Brand List
    heineken_brand_list = ["Heineken", "Tiger", "Bia Viet", "Larue", "Bivina", "Edelweiss", "Strongbow"]
    beer_competitor_list = ["Budweiser", "Bud Light", "Corona", "Miller Lite", "Coors Light", 
                            "Stella Artois", "Guinness", "Carlsberg", "Hoegaarden", "Chang", 
                            "Sapporo", "Asahi", "Saigon Beer", "Hanoi Beer", "Huda Beer", "Zorok"]
    soft_drink_competitor_list = ["Coca-Cola", "Pepsi", "7UP", "Sprite", "Fanta", 
                                  "Mountain Dew", "Dr Pepper", "Mirinda", "Schweppes", 
                                  "Red Bull", "Monster", "Sting", "Number 1"]
    mineral_water_competitor_list = ["Evian", "Perrier", "San Pellegrino", "Aquafina", "Dasani", 
                                    "Vittel", "Fiji", "Voss", "Poland Spring", "La Vie", 
                                    "Lavie", "Nestlé Pure Life", "Ice Mountain", "Crystal Geyser"]
                                    
    df_OCR = process_ocr_data(df_OCR, image)
    brand_class = []
    for _, obj in df_OD.iterrows():
        matched_brands = []
        for _, ocr in df_OCR.iterrows():
            if is_within(ocr, obj) and ocr['name'] is not None:
                matched_brands.append(ocr['name'])
        new_name = obj['name'] + ' ' + ' '.join(matched_brands) if matched_brands else None


        brand_class.append(new_name)
    df_OD['brand_class'] = brand_class

    covering_objects = ["signage", "billboard", "poster", "standee"]
    covered_objects = ["beer bottle", "beer can"]
    filtered_df = df_OD.copy()
    for _, cover in df_OD[df_OD['name'].isin(covering_objects)].iterrows():
        for _, covered in df_OD[df_OD['name'].isin(covered_objects)].iterrows():
            if is_within(covered, cover):
                filtered_df = filtered_df[(filtered_df['xmin'] != covered['xmin']) |
                                          (filtered_df['ymin'] != covered['ymin']) |
                                          (filtered_df['xmax'] != covered['xmax']) |
                                          (filtered_df['ymax'] != covered['ymax'])]
    filtered_df = filtered_df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name', 'brand_class']]



    final_df = pd.concat([filtered_df, df_OCR], axis=0)
    # Annotate images

    annotated_image = opencv_image.copy()

    for _, row in final_df.iterrows():
        if row['brand_class'] is None:
          cv2.rectangle(annotated_image, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 0, 255), 2)
          cv2.putText(annotated_image, row['name'], (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for _, row in final_df.iterrows():
        if row['brand_class'] is not None:
          cv2.rectangle(annotated_image, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 0, 0), 2)
          # cv2.putText(annotated_image, row['brand_class'], (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
          cv2.putText(annotated_image, str(row['brand_class']), (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return final_df, annotated_image

# Example usage
image_path = '/content/7.jpg'
df_OCR = paddle_OCR(image_path, Paddle_OCR)
# final_df, all_df, annotated_image_1, brand_df, annotated_image_2 = process_image(image_path, df_OCR)
final_df, annotated_image = process_image(image_path, df_OCR)

print(final_df)
import matplotlib.pyplot as plt
sv.plot_image(annotated_image, (10, 10))