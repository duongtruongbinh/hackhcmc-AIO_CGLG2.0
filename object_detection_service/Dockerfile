FROM wallies/python-cuda:3.9-cuda11.7-runtime

WORKDIR /app

COPY requirements.txt /app/
RUN pip install onnxruntime-gpu==1.15.1
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git
RUN pip install -q git+https://github.com/THU-MIG/yolov10.git
RUN pip -q install git+https://github.com/openai/CLIP.git
RUN pip install huggingface-hub
COPY . /app
EXPOSE 8002
CMD ["python", "api.py"]
