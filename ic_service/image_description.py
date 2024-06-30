# Main function

from langchain_core.messages import HumanMessage
from helper_functions import convert_df2str, convert
import time
# -----***-----
# Prompt list for 5 business problems
# Business problem 1: Count the number of people using beer products
problem_1 = """
1. Identification of people:
- Identify and describe all individuals in the image.
- Clearly state the number of individuals identified and their specific activities and emotions.
2. Confirmation of the customers use beer products:
- Identify and describe all individuals holding or near a beer bottle, can, or glass in the image.
- Clearly state the number of individuals identified and their specific activities and emotions.
"""

# Business problem 2: Detect advertising or promotional items from beer brands
problem_2 = """
1. Identify any logos present in the image:
- These logos may include text (with various typefaces/fonts), symbols, or a combination.
- Describe all items with the identified logo, providing details about the item's type, size, color, and appearance.
2. Identify advertisement or promotional items with identified logos in the image:
- Describe all advertisement or promotional items with the identified logo, such as refrigerators (or beverage coolers), advertising signs, posters, table standees, displays, standees, ice buckets, and parasols (if present).
Merge the same information and ignore duplicate information.
"""

# Business problem 3: Evaluating the success of the event
problem_3 = """
1. Identification of people:
- Identify and describe all individuals in the image.
- Clearly state the number of individuals identified and their specific activities and emotions.
2. Confirmation of the customers use beer products:
- Identify and describe all individuals holding or near a beer bottle, can, or glass in the image.
- Provide details about the beer product or nearby advertisement/marketing items including its appearance or brand logos (if present).
3. Crowd's emotion and activities recognition:
- Describe the overall activities and atmosphere of the crowd. Is it happy, angry, enjoyable, relaxed, neutral or something else?
"""

# Business problem 4: Track marketing staff
problem_4 = """
- Identify and describe marketing staffs who are standing, wearing branded uniforms, and involve in marketing activities present in the image.
- Provide details on their appearance, clothing, logo (if present), and any visible branding or promotional materials they are handling.
2. Confirmation of Staff Presence:
- Clearly state the number of marketing staff members identified in the image and their specific activities related to product promotion.
- Verify whether there are at least 2 marketing staff members present at the location.
Ensure that the analysis is thorough and accurate, focusing on confirming the presence and activities of the marketing staff.
"""

# Business issue 5: Assess the level of presence of beer brands in convenience stores/supermarkets
problem_5 = """
1. Identify any logos present in the image:
- The logos may include text (with various typefaces/fonts), symbols, or a combination.
- Describe all items with the identified logo, providing details about the item's type, size, color, and appearance.
2. Identify brand items and advertisement items with identified logos:
- Describe all packaging of brands, that have the identified logo.
- Describe all advertisement items with the identified logo, such as refrigerators (or beverage coolers), advertising signs, posters, table standees, standees, display stands, and parasols (if present).
Merge the same information and ignore duplicate information.
Comment on the overall presentation and organization of identified brand items or advertisement items in the image.
"""

# -----***-----
# Define the default user choice options (if no user-selected choice)
default_opts = {"convenience store": ["problem2", "problem4", "problem5"],
                "supermarket": ["problem2", "problem4", "problem5"],
                "bar or karaoke": ["problem2", "problem4"],
                "event": ["problem2", "problem3", "problem4"],
                "restaurant": ["problem1", "problem2", "problem3", "problem4"], }
# Define the business problem prompts
problem_prompts = {"problem1": problem_1,
                   "problem2": problem_2,
                   "problem3": problem_3,
                   "problem4": problem_4,
                   "problem5": problem_5}

# -----***-----
# -----***-----
# image description using only image


def create_description(prompt_text, img_str, vlm):
    img = convert(img_str)
    results = vlm.generate_content([prompt_text, img])
    return results.text

# -----***-----
# Image summarization


def image_summarizing(img_str, options, CLIP_class, YOLOw_df, vlm, llm):
    results = {}
    if len(options) == 0:
        choices = default_opts[CLIP_class]
    else:
        choices = options
    # Iterate over each choice in the list
    for choice in choices:
        problem_text = problem_prompts.get(choice, "")
        prompt_text = f"Combining the given image with extra detailed information as follows: \
        \nThis is a photo at the '{CLIP_class}'. \
        \nObjects and their coordinates in the image: \n{convert_df2str(YOLOw_df)}. \
        \nAnalysis and to perform the following tasks: \n{problem_text}"
        # image description using only image
        print("Start image description using only image")
        description = create_description(prompt_text, img_str, vlm)
        time.sleep(1)
        # image enhanced description using CLIP, PadleOCR, Owlv2, YOLOv10 combine with image description using only image
        # message = HumanMessage(
        #    content=[
        #        {
        #            "type": "text",
        #            "text": f"Given the initial description: '{description}' and detailed information as follows: \
        #        \nThis is a photo at the '{CLIP_class}'. \
        #        \nObjects and their coordinates in the image: \n{convert_df2str(YOLOw_df)}. \
        #        \nAdd detailed infomation to enhance initial description of the image. \
        #        \nNote that there may be duplicate information, remove it and only use available information."
        #        },
        #    ]
        # )
        # print("Start image enhanced description using CLIP, PadleOCR, Owlv2, YOLOv10 combine with image description using only image")
        # content = llm.invoke([message]).content
        # results[f'{choice}'] = content
        results[f'{choice}'] = description
    return results

# -----***-----
# Image hashtags extracting


def extract_hashtag(llm, description):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"Create a list of the top ten best-suited hashtags for the scene from the following image description: {description}. \
            Consider relevant themes, locations, and activities in the scene. \
            For example: Hashtags: [#bar, #pub, #restaurant, #conveniencestore, #supermarket, \
            #event, #party, #celebration, #gathering, #happyhour, #funtime.]"
            },
        ]
    )
    content = llm.invoke([message]).content
    if len(content.split('\n')) > 1:
        hashtags_list = [line.split('. ')[1].strip()
                         for line in content.split('\n')]
    else:
        hashtags_list = content.split('\n')
    return hashtags_list
