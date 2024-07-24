import torch
import clip
from PIL import Image
import base64
from io import BytesIO
import regex as re
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
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


def convert_image(input):
    image_data = base64.b64decode(input)
    image_data = BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image


def classify_image_clip(image_path):
    image = convert_image(image_path)
    # Function to tokenize and encode text descriptions

    def encode_text(text_descriptions):
        text_tokens = clip.tokenize(text_descriptions).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    # Function to process image and get top label
    def get_top_label(image_features, text_features, descriptions):
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, top_labels = text_probs.cpu().topk(1, dim=-1)
        return descriptions[top_labels[0][0].item()]

    # Process for location descriptions
    text_features = encode_text(scene_descriptions)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    location = get_top_label(image_features, text_features, scene_descriptions)
    location = re.sub(r'\ba photo at the\b', '', location).strip()

    # Process for environment descriptions
    text_features = encode_text(environment_descriptions)
    environment = get_top_label(
        image_features, text_features, environment_descriptions)
    environment = re.sub(r'\ba photo in the\b', '',
                         environment).strip().replace(' environment', '')

    return location, environment
