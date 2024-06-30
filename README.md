# Heineken's image analysis tool - AIO_CGLG2.0 <Br>

## Table of Contents <Br>
- Introduction
- Features
- Installation
- Usage
- Examples

### Introduction <Br>
Our project focuses on enhancing information extraction from an image by integrating Object Detection and Image Captioning. User input image which will be location classified first, then suggest the relevant problems and output bounding boxes of objects and a descriptive image caption. Beside that, we also have a unique feature allows users customize the problems, tailoring the detection and captioning processes to their specific concerns. <Br>
Aim to develop an image analysis tool that can automatically detect the following factors:
- **Brand logos**: Detect logos of Heineken, Tiger, Bia Viet, Larue, Bivina, Edelweiss and Strongbow.
- **Products**: Identify boxes of beer and beer bottles.
- **Consumers**: Evaluate the number, activities and emotions of customers.
- **Advertising items**: Identify brand posters, banners and billboards
- **Visual Context**: Analysis of the location—restaurant, bar, grocery store or supermarket, etc...

### Features
![Imgur](https://i.imgur.com/rTHa3jj.png)

List the main features of the project:
- **Location classification**: CLIP (Contrastive Language-Image Pre-Training)
- **Object detection**: PaddleOCR, Owlv2 (Open-World Localization v2), and YOLOv10 
- **Image captioning**: Gemini-1.5-pro-lastest
- **Enhance information extraction**: Gemini-pro

Anothe technical features:
- **APIs**: FastAPI, Streamlit
- **Deployment**: ngrok

### Installation
1. Clone the repository
```bash
git clone https://github.com/duongtruongbinh/hackhcmc-AIO_CGLG2.0
```
2. Install the required packages
```bash
pip install -r requirements.txt
```
### Usage

#### Set up .env files
Create a .env file in each microservice folder and add the following variables (customize for each microservice):
```bash
OCR_URL = <local or public URL of the OCR service>/process_image
IC_URL = <local or public URL of the IC service>/image_captioning
CLIP_URL = <local or public URL of the CLIP service>/image_classifier
OD_URL = <local or public URL of the OD service>/object_detection
MULTI_URL = <local or public URL of the MULTI service>/multi_image
GEMINI_KEY = <API key of the Gemini service>
```
#### Run each microservice separately:

1. Location classification
```bash
cd clip_service
python api.py
```
2. Object detection
```bash
cd od_service
python api.py
```
3. Image captioning
```bash
cd ic_service
python api.py
```
4. PaddleOCR
```bash
cd PaddleOCR/ppstructure
python api.py
```
#### Run app
```bash
cd ic_service
streamlit run app.py
```

### Setup and public demo API via ngrok:
1. Install ngrok

2. Add authentication token
```bash
ngrok config add-authtoken <your_auth_token>
```
3. Run ngrok
```bash
ngrok http <path_streamlit_web_app>
```
### Team members

- Minh-Dung Le-Quynh
- Truong-Binh Duong
- Thuy-Ha Le-Thi
- Binh-Nam Le-Nguyen

