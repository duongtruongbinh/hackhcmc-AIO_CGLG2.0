import cv2
import supervision as sv
from ultralytics import YOLO
import os
import base64
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import textdistance
from sklearn.cluster import DBSCAN
import torch
from torchvision.ops import nms
from PIL import Image
from io import BytesIO
pd.options.mode.copy_on_write = True
heineken_brand_list = ["Heineken", "Tiger", "Bia Viet",
                       "Larue", "Bivina", "Edelweiss", "Strongbow"]
beer_competitor_list = ["Budweiser", "Bud Light", "Corona", "Miller Lite", "Coors Light",
                        "Stella Artois", "Guinness", "Carlsberg", "Hoegaarden", "Chang",
                        "Sapporo", "Asahi", "Saigon Beer", "Hanoi Beer", "Huda Beer", "Zorok"]
soft_drink_competitor_list = ["Coca-Cola", "Pepsi", "7UP", "Sprite", "Fanta",
                              "Mountain Dew", "Dr Pepper", "Mirinda", "Schweppes",
                              "Red Bull", "Monster", "Sting", "Number 1"]
mineral_water_competitor_list = ["Evian", "Perrier", "San Pellegrino", "Aquafina", "Dasani",
                                 "Vittel", "Fiji", "Voss", "Poland Spring", "La Vie",
                                 "Lavie", "Nestlé Pure Life", "Ice Mountain", "Crystal Geyser"]


def convert_OCR_result(results):
    if len(results[0]) == 0:
        return None
    else:
        new_results = []
        for result in results[0]:
            f_points, (label, score) = result
            points = np.array(f_points, np.int32)
            # Calculate xmax, ymax, xmin, ymin
            x_coords = [coord[0] for coord in points]
            y_coords = [coord[1] for coord in points]
            xmax = max(x_coords)
            xmin = min(x_coords)
            ymax = max(y_coords)
            ymin = min(y_coords)
            # Arrange into [xmin, ymin, xmax, ymax] format
            bbox = [xmin, ymin, xmax, ymax]
            new_results.append((bbox, score, label))
        # Create DataFrame directly
        new_results_df = pd.DataFrame([(box[0], box[1], box[2], box[3], conf, label) for box, conf, label in new_results],
                                      columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name'])
        return new_results_df


def load_image(image_file):
    image_data = base64.b64decode(image_file)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img


def filter_detections(detections, class_filter, confidence_threshold):
    filtered_indices = (detections.class_id == class_filter) & (
        detections.confidence >= confidence_threshold
    )
    return sv.Detections(
        xyxy=detections.xyxy[filtered_indices],
        confidence=detections.confidence[filtered_indices],
        class_id=detections.class_id[filtered_indices],
    )


def annotate_image(image, detections, class_names):
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{class_names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(
            detections.class_id, detections.confidence
        )
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )
    return annotated_image


def map_ocr_to_brands(df_OCR, threshold=0.3):
    def find_best_match(text):
        best_match, best_score = None, float('inf')
        all_brands = heineken_brand_list + beer_competitor_list + \
            soft_drink_competitor_list + mineral_water_competitor_list
        for brand in all_brands:
            distance = textdistance.levenshtein.normalized_distance(
                text.lower(), brand.lower())
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
    image = np.array(image)
    image_height, image_width = image.shape[:2]
    eps = max(image_width / 5, image_height / 5)
    return eps


def find_groups_dbscan(rectangles, image, min_samples=4):
    if rectangles.empty:
        return rectangles
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

    if df_OCR.empty:
        df_OCR["brand_class"] = pd.Series(dtype='object')
        return df_OCR

    # Calculate the number of rows in each group
    group_sizes = df_OCR.groupby('group').size().reset_index(name='group_size')

    # Merge group sizes into the original dataframe
    df_OCR = df_OCR.merge(group_sizes, on='group', how='left')

    # Determine brand_class based on the number of rows in the group
    def determine_brand_class(row):
        if row['group_size'] == 1:
            return f"{row['new_name']} logo"
        elif row['group_size'] > 2:
            return f"{row['new_name']} beer carton"
        return None

    df_OCR['brand_class'] = df_OCR.apply(determine_brand_class, axis=1)

    # Filter out rows with null brand_class and create a copy to avoid SettingWithCopyWarning
    df_OCR = df_OCR[df_OCR['brand_class'].notna()].copy()

    # Select only the required columns
    df_OCR = df_OCR[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name', 'brand_class']]

    return df_OCR


def detect_person(image):
    model = YOLO(os.path.join("weight", 'yolov10x.pt'))
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
    # Convert the 'name' column to string type
    results_df['name'] = results_df['name'].astype(str)
    # Filter the DataFrame and use .loc to modify the 'name' column
    results_df = results_df[results_df['name'] == 0].copy()
    results_df.loc[:, 'name'] = 'person'

    return results_df

# Object Detection Function


def owlv2_detection(im):
    checkpoint = "google/owlv2-base-patch16-ensemble"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)

    text_queries = ["beer bottle", "beer can", "beer carton", "beer crate", "ice bucket", "ice box", "fridge",
                    "signage", "billboard", "poster", "standee", "tent card", "display stand", "tabletop display", "parasol"]
    thresholds = {"beer bottle": 0.3, "beer can": 0.3, "beer carton": 0.4, "beer crate": 0.4, "ice bucket": 0.4, "ice box": 0.4, "fridge": 0.4,
                  "signage": 0.4, "billboard": 0.3, "poster": 0.3, "standee": 0.3, "tent card": 0.4, "display stand": 0.4, "tabletop display": 0.4, "parasol": 0.4}

    inputs = processor(text=text_queries, images=im, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

    width = target_sizes[0][0].item()
    height = target_sizes[0][1].item()
    height_ratio = height / width if width > height else 1
    width_ratio = width / height if height > width else 1
    width_ratio, height_ratio = height_ratio, width_ratio

    scores = results["scores"].clone().detach()
    labels = results["labels"].clone().detach()
    boxes = results["boxes"].clone().detach()

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
            data["xmin"].append(int(box[0] / width_ratio))
            data["ymin"].append(int(box[1] / height_ratio))
            data["xmax"].append(int(box[2] / width_ratio))
            data["ymax"].append(int(box[3] / height_ratio))
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
    ratio = inter_area / (box1_area + 1e-6)

    # Check if the ratio exceeds the threshold
    return ratio >= threshold

# Main function to process and create the final dataframe


def convert_image(input):
    image_data = base64.b64decode(input)
    image_data = BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image


def process_image(image_path, df_OCR):
    path_copy = image_path
    im = convert_image(path_copy)
    df_owlv2 = owlv2_detection(im)

    # Create a set to keep track of indices to drop
    indices_to_drop = set()

    for i, box1 in df_owlv2.iterrows():
        for j, box2 in df_owlv2.iterrows():
            if i != j and is_within(box1, box2, threshold=1.0):
                if (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin']) < (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin']):
                    indices_to_drop.add(i)
                else:
                    indices_to_drop.add(j)

    df_owlv2 = df_owlv2.drop(indices_to_drop).copy()

    df_person = detect_person(im)

    df_OD = pd.concat([df_owlv2, df_person], axis=0)

    df_OCR = process_ocr_data(df_OCR, im)
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

    filtered_df = filtered_df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name', 'brand_class']].copy()

    final_df = pd.concat([filtered_df, df_OCR], axis=0).copy()
    all_count_df = pd.DataFrame(final_df.groupby('name').size().reset_index(name='count'))
    all_count_df.rename(columns={'name': 'object'}, inplace=True)

    ob_list = ["person", "beer bottle", "beer can", "beer carton", "beer crate", "ice bucket", "ice box", "fridge",
               "signage", "billboard", "poster", "standee", "tent card", "display stand", "tabletop display", "parasol"]
    all_count_df = all_count_df[all_count_df['object'].isin(ob_list)].copy()

    sub_brand_count_df = pd.DataFrame(final_df.groupby('brand_class').size().reset_index(name='count')).copy()

    heineken_brand_count_df = sub_brand_count_df[~sub_brand_count_df['brand_class'].str.contains('competitor')].copy()
    heineken_brand_count_df.rename(columns={'brand_class': 'heineken_object'}, inplace=True)

    competitor_brand_count_df = sub_brand_count_df[sub_brand_count_df['brand_class'].str.contains('competitor')].copy()
    competitor_brand_count_df.rename(columns={'brand_class': 'competitor_object'}, inplace=True)

    beer_number = all_count_df[all_count_df['object'].isin(['beer can', 'beer bottle'])]['count'].sum()
    person_number = all_count_df[all_count_df['object'] == 'person']['count'].sum()
    promo_person_number = heineken_brand_count_df[heineken_brand_count_df['heineken_object'].str.contains('person')]['count'].sum()

    if beer_number >= (person_number - promo_person_number):
        drinking_person = person_number - promo_person_number
    else:
        drinking_person = beer_number

    if drinking_person != 0:
        df_drinking_person = pd.DataFrame({'object': ['drinking person'], 'count': [drinking_person]})
        all_count_df = pd.concat([all_count_df, df_drinking_person], ignore_index=True)

    return final_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df
