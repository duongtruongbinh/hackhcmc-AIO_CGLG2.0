import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from ast import literal_eval
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
if 'analysis_results' not in st.session_state:
    st.session_state.analyzed_image = None

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
    st.write("This feature is under development.")
    # Allow users to upload multiple files
    uploaded_files = st.file_uploader(
        "Choose image files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    # If files are uploaded
    if uploaded_files:
        st.write("Number of uploaded images:", len(uploaded_files))

        analysis_results = []  # To store the results for Excel export
        # print(uploaded_files)
        # Display and analyze each image
        file_paths = [uploaded_file.name for uploaded_file in uploaded_files]
        multi_url = os.environ.get("MULTI_URL")
        problem_dict = {
            "problem1": "Problem 1: Count the number of people using beer products",
            "problem2": "Problem 2: Detect advertising or promotional items from beer brands",
            "problem3": "Problem 3: Evaluating the success of the event",
            "problem4": "Problem 4: Track marketing staff",
            "problem5": "Problem 5: Assess the level of presence of beer brands in convenience stores/supermarkets"
        }

        all_problems = list(problem_dict.values())
        problems = st.multiselect(
            "Select the problems to analyze", all_problems, default=problem_dict["problem1"])
        
        base64_list = [convert_image_to_base64(uploaded_file) for uploaded_file in uploaded_files]
        options = [key for key, value in problem_dict.items() if value in problems]
        myobj = {
            "img_base64_list": base64_list,
            "options": options
        }
        start_time = time.time()
        with st.spinner("Analyzing images... Please wait."):  
            x = requests.post(multi_url, json=myobj)
            y = json.loads(x.text)
            scene_hashtag_list, content = literal_eval(y['scene_hashtag_list']), y['content']
        end_time = time.time()
        # show images
        n_col = 2
        n_row = len(uploaded_files) // n_col
        if len(uploaded_files) % n_col != 0:
            n_row += 1
        cols = st.columns(n_col)
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % n_col]:
                image = Image.open(uploaded_file)
                st.image(image.resize((image.width // 2, image.height // 2)))
                
                for s in scene_hashtag_list[i]:
                    
                    st.markdown(s, unsafe_allow_html=True)
        st.markdown("## Analysis Results")
        st.markdown(content)
        st.markdown(f"Elapsed time: {end_time - start_time:.2f} seconds")
        


def single_image():
    st.write("Upload an image and choose the problem to analyze.")
    uploaded_file = st.file_uploader(
        "Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image_base64 = convert_image_to_base64(uploaded_file)
        default_opts = {
            "convenience store": ["problem2", "problem4", "problem5"],
            "supermarket": ["problem2", "problem4", "problem5"],
            "bar or karaoke": ["problem2", "problem4"],
            "event": ["problem2", "problem3", "problem4"],
            "restaurant": ["problem1", "problem2", "problem3", "problem4"]
        }
        cols_first = st.columns(2)
        with cols_first[0]:
            image = Image.open(uploaded_file)
            st.image(image.resize((image.width // 2, image.height // 2)))

        location, environment = clip_api(uploaded_file)
        with cols_first[1]:
            st.markdown(f"""**Location**: #{location}<br>
                            **Environment**: #{environment}<br>""", unsafe_allow_html=True)
        problem_dict = {
            "problem1": "Problem 1: Count the number of people using beer products",
            "problem2": "Problem 2: Detect advertising or promotional items from beer brands",
            "problem3": "Problem 3: Evaluating the success of the event",
            "problem4": "Problem 4: Track marketing staff",
            "problem5": "Problem 5: Assess the level of presence of beer brands in convenience stores/supermarkets"
        }

        all_problems = list(problem_dict.values())
        problems = st.multiselect(
            "Select the problems to analyze", all_problems,
            default=[problem_dict[problem]
                     for problem in default_opts[location]])

        if "analyzed_image" not in st.session_state or st.session_state.analyzed_image != image_base64:
            st.session_state.analyzed_image = image_base64
            st.session_state.location = location
            st.session_state.all_problems = all_problems

            start_time = time.time()
            with st.spinner("Analyzing image... Please wait."):
                results = analyze_image(uploaded_file, location, all_problems)
            end_time = time.time()

            st.session_state.analysis_results = results
            st.session_state.elapsed_time = end_time - start_time

        if "analyzed_image" in st.session_state:
            if st.session_state.analyzed_image == image_base64:
                (scene_hastags, enhance_description, yolo_df, all_count_df,
                 heineken_brand_count_df, competitor_brand_count_df) = st.session_state.analysis_results

                scene_hastags = ", ".join(scene_hastags)
                yolo_df = pd.DataFrame(yolo_df)
                with cols_first[1]:
                    st.markdown(
                        f"""**Scene Hashtags**: {scene_hastags}""", unsafe_allow_html=True)

                    show_dfs = []
                    if all_count_df is not None:
                        all_count_df = pd.DataFrame(all_count_df)
                        if location != "restaurant":
                            all_count_df = all_count_df[all_count_df['object']
                                                        != "drinking person"]
                        show_dfs.append(all_count_df)

                    if heineken_brand_count_df is not None:
                        heineken_brand_count_df = pd.DataFrame(
                            heineken_brand_count_df)
                        show_dfs.append(heineken_brand_count_df)
                    if competitor_brand_count_df is not None:
                        competitor_brand_count_df = pd.DataFrame(
                            competitor_brand_count_df)
                        show_dfs.append(competitor_brand_count_df)

                    # number col = num of show_dfs
                    cols = st.columns(len(show_dfs))
                    for i, df in enumerate(show_dfs):
                        with cols[i]:
                            st.dataframe(df)

                for problem in problems:
                    problem_key = [
                        key for key, value in problem_dict.items() if value == problem][0]
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if yolo_df.empty:
                            annotated = image
                        else:
                            annotated = draw_annotated(
                                problem_key, yolo_df, image)
                        st.image(annotated)
                    with cols[1]:
                        st.markdown(problem)
                        st.markdown(enhance_description[problem])

                st.write(
                    f"Elapsed time: {st.session_state.elapsed_time:.2f} seconds")


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
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.0005*image.height, (255, 0, 0), image.height // 300)

    if problem in ["problem2", "problem5"]:
        # Vẽ khung chữ nhật cho các đối tượng không phải 'person'
        for index, row in non_persons.iterrows():
            x, y, w, h = int(row["xmin"]), int(row["ymin"]), int(
                row["xmax"] - row["xmin"]), int(row["ymax"] - row["ymin"])
            annotated = cv2.rectangle(
                annotated, (x, y), (x + w, y + h), (255, 0, 0), thickness=image.height // 200)
            annotated = cv2.putText(
                annotated, row["name"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.0005*image.height, (255, 0, 0), image.height // 300)
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

    myobj = {
        "base64_string": convert_image_to_base64(upload_file),
        "location": location,
        "options": options
    }
    ic_url = os.environ.get("IC_URL")
    x = requests.post(ic_url, json=myobj)
    y = json.loads(x.text)
    
    try:
        scene_hashtags = y["scene_hashtags"]
        enhanced_description = y["enhanced_description"]
        yolo_df = literal_eval(y["yolo_df"])
        all_count_df = literal_eval(y["all_count_df"])
        heineken_brand_count_df = literal_eval(y["heineken_brand_count_df"])
        competitor_brand_count_df = literal_eval(y["competitor_brand_count_df"])
        print(yolo_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df)
    except (KeyError, ValueError) as e:
        print(f"Error processing the response: {e}")
        return None, None, None, None, None, None
    
    return scene_hashtags, enhanced_description, yolo_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df


def show_info():
    # Display app information
    st.write("Application Information")

    st.write("""
        This application is created to analyze images for Heineken Vietnam.
        You can upload image files, analyze them, and view the overall results.
    """)


if __name__ == "__main__":
    main()
