from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from io import BytesIO
import torch
import requests
import io

app = FastAPI()

"""
 物体検出API
"""

# ヘルスチェック


@app.get("/")
def read_root():
    return {"Status": "OK"}

# 物体検出


@app.post("/detect/")
def detect(img: UploadFile = File(...), threshold: float = 0.25):
    # 画像のロード
    image = Image.open(img.file)
    image = image.convert("RGB")

    # 予測と描画
    image = _object_detection(image, threshold)

    response = BytesIO()
    image.save(response, "JPEG")
    response.seek(0)

    return StreamingResponse(response, media_type="image/jpeg")


def _object_detection(image, threshold):
    # 物体検出
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    pred = model(image)

    # 色の一覧を作成
    cmap = plt.cm.get_cmap("hsv", len(model.model.names))

    # フォント設定
    truetype_url = 'https://github.com/JotJunior/PHP-Boleto-ZF2/blob/master/public/assets/fonts/arial.ttf?raw=true'
    r = requests.get(truetype_url, allow_redirects=True)
    size = int(image.size[0]*0.02)
    font = ImageFont.truetype(io.BytesIO(r.content), size=size)

    # 検出結果の描画
    for detections in pred.xyxy:
        for detection in detections:
            class_id = int(detection[5])
            class_name = str(model.model.names[class_id])
            bbox = [int(x) for x in detection[:4].tolist()]
            conf = float(detection[4])
            # 閾値以上のconfidenceの場合のみ描画
            if conf >= threshold:
                color = cmap(class_id, bytes=True)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline=color, width=3)
                draw.text([bbox[0]+5, bbox[1]+10], class_name, fill=color, font=font)

    return image
