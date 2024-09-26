from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import pandas as pd
import base64
import os

# Format Output for each image: 
location_list = ['convenience store','supermarket','restaurant','event','bar', 'karaoke']
object_list = ["person", "promotional person", "beer bottle", "beer can", "beer carton", "beer crate",
		        "ice bucket", "ice box", "fridge",
                "signage", "billboard", "poster", "standee", "tent card", 
		        "display stand", "tabletop display", "parasol"]
brand_list = ['Heineken','Tiger','Bia Viet','Larue', 'Bivina', 'Edelweiss','Strongbow']
context_prompt = f'''
Analyze the image encoded in base64 and return the following information:
- obj_detects: List of ObjDetect. Each ObjDetect is just for 1 object in the provided object list = {object_list}
- ObjDetect information:
  + object_name: Identify the object in the image according to the provided object list = {object_list}. Please use exactly name on the list.
  + brand_objs: List of BrandObj related to this object_name in the image. Each BrandObj is just for 1 brand in the provided brand list = {brand_list}.
- BrandObj information:
  + brand_name: Detect the brand name of the object based on the provided brand list = {brand_list} (if the brand logo is visible). If the brand is not in the list, specify the name of that new brand.
  + brand_type: Choose one type in list "heineken brand", "beer competitor", "soft drink competitor" and "mineral water competitor". This type should be related to this brand_name.
  + brand_object_count: Count the number of the object with this brand_name in the image.
'''

# - short_description: Provide a brief description of the image in less than 15 words.
# - emotion: Identify the emotion of any person in the image, choosing from four options: 'not interested', 'neutral', 'happy', or 'null' (if no person is present).
# - Is new brand: Indicate whether the brand is new (true if the brand is not in the brand list = {brand_list}, false if it is).

class ImageInfo(BaseModel):
    # short_description: str
    # emotion: str
    class ObjDetect(BaseModel):
        object_name: str
        class BrandObj(BaseModel):
            brand_name: str
            # is_new_brand: bool
            brand_type: str
            brand_object_count: int
        brand_objs: list[BrandObj]
    obj_detects: list[ObjDetect]

def detect_obj_openai(img_str, gpt):
    model = 'gpt-4o-2024-08-06'
    completion = gpt.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img_str}"
                                        }
                                    }
                                ]
            }
        ],
        response_format=ImageInfo,
        # max_tokens=500
    )

    image_info = completion.choices[0].message

    if (image_info.refusal):
        print(f"{image_info.refusal}")
        return pd.DataFrame(columns=['heineken_object', 'count']), pd.DataFrame(columns=['competitor_object', 'count'])
    else:
        image_result = image_info.parsed.dict()
        heineken_data = []
        competitor_data = []

        # Loop through detected objects
        for obj in image_result['obj_detects']:
            object_name = obj['object_name']
            for brand in obj['brand_objs']:
                brand_name = brand['brand_name']
                brand_type = brand['brand_type']
                brand_count = brand['brand_object_count']

                # Check if it's Heineken or competitor
                if brand_type == "heineken brand":
                    heineken_data.append({
                        'heineken_object': f"{brand_name} {object_name}",
                        'count': brand_count
                    })
                else:
                    competitor_data.append({
                        'competitor_object': f"({brand_type}) {brand_name} {object_name}",
                        'count': brand_count
                    })

        # Create DataFrames
        heineken_brand_count_df = pd.DataFrame(heineken_data, columns=['heineken_object', 'count'])
        competitor_brand_count_df = pd.DataFrame(competitor_data, columns=['competitor_object', 'count'])

        return heineken_brand_count_df, competitor_brand_count_df


def combine_detect_df(heineken_brand_count_df_1, competitor_brand_count_df_1, heineken_brand_count_df_2, competitor_brand_count_df_2):
    # Combine Heineken DataFrames
    heineken_combined_df = pd.concat([heineken_brand_count_df_1, heineken_brand_count_df_2])
    heineken_combined_df = heineken_combined_df.groupby('heineken_object', as_index=False).agg({'count': 'max'})

    # Combine Competitor DataFrames
    competitor_combined_df = pd.concat([competitor_brand_count_df_1, competitor_brand_count_df_2])
    competitor_combined_df = competitor_combined_df.groupby('competitor_object', as_index=False).agg({'count': 'max'})

    return heineken_combined_df, competitor_combined_df

if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    gpt = OpenAI(api_key=OPENAI_API_KEY)


    # Load and encode the image
    image_path = 'test_image/test.jpg'  # Replace with your actual image path
    with open(image_path, "rb") as image_file:
        img_str = base64.b64encode(image_file.read()).decode('utf-8')

    # Detect objects and brands
    heineken_brand_count_df_1, competitor_brand_count_df_1 = detect_obj_openai(img_str, gpt)
    # Output results
    print()
    print("**************************")
    print()
    print("Heineken Brand Counts:")
    print(heineken_brand_count_df_1)
    print("\nCompetitor Brand Counts:")
    print(competitor_brand_count_df_1)

    # Create temporary DataFrames for testing (these can be results from another image)
    data_heineken_2 = [{'heineken_object': 'Tiger beer carton', 'count': 5},
                       {'heineken_object': 'Heineken person', 'count': 3}]
    data_competitor_2 = [{'competitor_object': '(beer competitor) Budweiser beer bottle', 'count': 10},
                         {'competitor_object': '(soft drink competitor) Mirinda beer carton', 'count': 5}]

    heineken_brand_count_df_2 = pd.DataFrame(data_heineken_2, columns=['heineken_object', 'count'])
    competitor_brand_count_df_2 = pd.DataFrame(data_competitor_2, columns=['competitor_object', 'count'])
    print()
    print("**************************")
    print()
    print("Temporary Heineken Brand Counts 2:")
    print(heineken_brand_count_df_2)
    print("\nTemporary Competitor Brand Counts 2:")
    print(competitor_brand_count_df_2)

    # Combine DataFrames
    heineken_brand_count_df, competitor_brand_count_df = combine_detect_df(
        heineken_brand_count_df_1, competitor_brand_count_df_1, heineken_brand_count_df_2, competitor_brand_count_df_2)

    # Output results
    print()
    print("**************************")
    print()
    print("Combined Heineken Brand Counts:")
    print(heineken_brand_count_df)
    print("\nCombined Competitor Brand Counts:")
    print(competitor_brand_count_df)


