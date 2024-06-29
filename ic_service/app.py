import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import ast
import numpy as np
import pandas as pd
import requests
import base64
from io import BytesIO
import json
from dotenv import load_dotenv
import time
import cv2
load_dotenv()
# Set environment variable to resolve OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


scene_descriptions = ["a photo at the convenience store",
                      "a photo at the supermarket",
                      "a photo at the restaurant",
                      "a photo at the event",
                      "a photo at the bar or karaoke",
                      ]
environment_descriptions = [
    "a photo in the outdoor environment",
    "a photo in the indoor environment",
]
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
# convert image to base64 string


def convert_image_to_base64(upload_file):
    file_format = "JPEG" if upload_file.name.endswith(
        "jpg") else upload_file.name.split(".")[-1].upper()
    image = Image.open(upload_file)
    buffered = BytesIO()
    image.save(buffered, format=file_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def main():
    st.set_page_config(
        page_title="Heineken Vietnam Auto Image Analysis",
        page_icon="🍺",
        layout="wide",  # Use wide layout for more space
    )
    st.title("Heineken Vietnam Auto Image Analysis")

    # Create sidebar
    with st.sidebar:
        selected = option_menu('Menu',
                               ['Multiple Images', 'Single Image', 'Info'],
                               menu_icon='house',
                               # Icons for each option
                               icons=['image', 'image', 'info-circle'],
                               default_index=0
                               )

    # Create main content
    with st.container():
        if selected == 'Multiple Images':
            multiple_images()
        elif selected == 'Single Image':
            single_image()
        elif selected == 'Info':
            show_info()


def multiple_images():
    st.write("""
        Upload a set of image files from a folder.
        You can select multiple files at once.
    """)
    # Allow users to upload multiple files
    uploaded_files = st.file_uploader(
        "Choose image files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    # If files are uploaded
    if uploaded_files:
        st.write("Number of uploaded images:", len(uploaded_files))

        analysis_results = []  # To store the results for Excel export

        # Display and analyze each image
        num_images = len(uploaded_files)
        multi_url = os.environ.get("MULTI_URL")
        myobj = {
            "img_path": uploaded_files,
            "options": ["problem1", "problem2", "problem3", "problem4", "problem5"]
        }
        x = requests.post(multi_url, json=myobj)
        y = json.loads(x.text)
        temp = y["context"]
        print(temp)
        for i in range(0, num_images, 3):  # Process 3 images at a time
            cols = st.columns(3)
            for j in range(3):
                if i + j < num_images:
                    with cols[j]:
                        uploaded_file = uploaded_files[i + j]
                        image = Image.open(uploaded_file)
                        # Resize image to a fixed size for better layout
                        image = image.resize((300, 300))
                        st.image(
                            image, caption=f"Image {i + j + 1}", use_column_width=True)


def single_image():
    st.write("Upload an image and choose the problem to analyze.")
    uploaded_file = st.file_uploader(
        "Choose an image file", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        default_opts = {"convenience store": ["problem2", "problem4", "problem5"],
                        "supermarket": ["problem2", "problem4", "problem5"],
                        "bar or karaoke": ["problem2", "problem4"],
                        "event": ["problem2", "problem3", "problem4"],
                        "restaurant": ["problem1", "problem2", "problem3", "problem4"]}

        image = Image.open(uploaded_file)

        st.image(image, use_column_width=True)
        location, environment = clip_api(uploaded_file)
        problem_dict = {
            "problem1": "Problem 1: Count the number of people using beer products",
            "problem2": "Problem 2: Detect advertising or promotional items from beer brands",
            "problem3": "Problem 3: Evaluating the success of the event",
            "problem4": "Problem 4: Track marketing staff",
            "problem5": "Problem 5: Assess the level of presence of beer brands in convenience stores/supermarkets"
        }
        problems = st.multiselect(
            "Select the problems to analyze", problem_dict.values(), default=[problem_dict[problem]
                                                                              for problem in default_opts[location]])
        start_time = time.time()
        scene_hastags, enhance_description, yolo_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df = analyze_image(
            uploaded_file, location, problems)
        end_time = time.time()
        eslap_time = end_time - start_time

        scene_hastags = ", ".join(scene_hastags)
        yolo_df = pd.DataFrame(yolo_df)
        st.markdown(f"""**Location**: #{location}<br>
                    **Environment**: #{environment}<br>
                    **Scene Hashtags**: {scene_hastags}<br>""", unsafe_allow_html=True)
        show_dfs = []
        if all_count_df is not None:
            show_dfs.append(all_count_df)
        if heineken_brand_count_df is not None:
            show_dfs.append(heineken_brand_count_df)
        if competitor_brand_count_df is not None:
            show_dfs.append(competitor_brand_count_df)

        cols = st.columns(len(show_dfs))
        for col, df in zip(cols, show_dfs):
            col.dataframe(df)
        for problem in problems:
            # take problem key from problem_dict
            problem_key = [key for key, value in problem_dict.items(
            ) if value == problem][0]
            cols = st.columns([1, 3])
            with cols[0]:
                if yolo_df.empty:
                    annotated = image
                else:
                    annotated = draw_annotated(problem_key, yolo_df, image)
                st.image(annotated)
            with cols[1]:
                st.markdown(problem)
                st.markdown(enhance_description[problem])

        # small text for time
        st.write(f"Elapsed time: {eslap_time:.2f} seconds")


# def count_and_check(yolo_df, problem):
#     result_dict = {}
#     persons = yolo_df[yolo_df["name"] == "person"]
#     non_persons = yolo_df[yolo_df["name"] != "person"]
#     if problem in ["problem1", "problem3"]:
#         # dem nguoi
#         result_dict["person"] = len(persons)
#     if problem in ["problem2", "problem5"]:
#         # dem vat pham thuoc heineiken
#         try:
#             result_dict.update(non_persons.value_counts(
#                 "branch_class").to_dict())
#             # dem competitor
#             result_dict.update(non_persons["competitor" in non_persons["branch_class"]].value_counts(
#                 "branch_class").to_dict())
#         except:
#             result_dict["competitor"] = 0
#             result_dict["heineken"] = 0

#     return result_dict


def draw_annotated(problem, yolo_df, image):
    persons = yolo_df[yolo_df["name"] == "person"]
    non_persons = yolo_df[yolo_df["name"] != "person"]
    annotated = np.array(image.copy())

    if problem in ["problem1", "problem3", "problem4"]:
        # Vẽ khung chữ nhật cho các đối tượng 'person'
        for index, row in persons.iterrows():
            x, y, w, h = int(row["xmin"]), int(row["ymin"]), int(
                row["xmax"] - row["xmin"]), int(row["ymax"] - row["ymin"])
            annotated = cv2.rectangle(
                annotated, (x, y), (x + w, y + h), (255, 0, 0), thickness=image.height // 200)
            annotated = cv2.putText(annotated, "Person", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.0008*image.height, (255, 0, 0), 5)

    if problem in ["problem2", "problem5"]:
        # Vẽ khung chữ nhật cho các đối tượng không phải 'person'
        for index, row in non_persons.iterrows():
            x, y, w, h = int(row["xmin"]), int(row["ymin"]), int(
                row["xmax"] - row["xmin"]), int(row["ymax"] - row["ymin"])
            annotated = cv2.rectangle(
                annotated, (x, y), (x + w, y + h), (255, 0, 0), thickness=image.height // 200)
            annotated = cv2.putText(
                annotated, row["name"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.0008*image.height, (255, 0, 0), 5)
    return annotated


def clip_api(upload_file):
    clip_url = os.environ.get("CLIP_URL")
    myobj = {
        "base64_string": convert_image_to_base64(upload_file)
    }
    x = requests.post(clip_url, json=myobj)
    y = json.loads(x.text)
    location, environment = y["location"], y["environment"]
    return location, environment


def analyze_image(upload_file, location, options):
    # Analyze an image
    ic_url = 'http://localhost:8003/image_captioning/'
    myobj = {
        "base64_string": convert_image_to_base64(upload_file),
        "location": location,
        "options": options
    }
    x = requests.post(ic_url, json=myobj)
    y = json.loads(x.text)
    temp = ast.literal_eval(y["ic"])
    scence_hastags, enhance_description, yolo_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df = temp
    return scence_hastags, enhance_description, yolo_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df


def show_info():
    # Display app information
    st.write("Application Information")

    st.write("""
        This application is created to analyze images for Heineken Vietnam.
        You can upload image files, analyze them, and view the overall results.
    """)


if __name__ == "__main__":
    main()
