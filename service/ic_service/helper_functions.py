# Helper function
import pandas as pd
from IPython.display import display
from IPython.display import Markdown
import textwrap
import PIL
from PIL import Image
import base64
from io import BytesIO

# convert img_str to image


def convert(base64_str):
    image_data = base64.b64decode(base64_str)
    image = BytesIO(image_data)
    img = Image.open(image)
    img.save("input.jpg")
    return img

# display text (markdown)


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# display image


def display_image(img_path, scale_down=2):
    img = PIL.Image.open(img_path).convert(mode="RGB")
    w, h = img.size
    display(img.resize((w//scale_down, h//scale_down)))

# convert dataframe to string


def convert_df2str(df):
    if isinstance(df, pd.DataFrame):
        string = "\n".join(
            [f'{index}: {[{col: row[col]} for col in df.columns]}' for index, row in df.iterrows()])
        return string
    else:
        return 'None'
