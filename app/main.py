from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from io import BytesIO
import torch

app = FastAPI()

"""
 物体検出
"""
@app.post("/detect/")
async def detect(img: UploadFile = File(...), threshold :float = 0.25):
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
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    pred = model(image)

    # 検出結果の描画
    for detections in pred.xyxy:
        # 色の一覧を作成
        cmap = plt.cm.get_cmap("hsv", len(detections) + 1)

        for i, detection in enumerate(detections):
            class_name = str(model.model.names[int(detection[5])])
            bbox = [int(x) for x in detection[:4].tolist()]
            conf = float(detection[4])
            # 閾値以上のconfidenceの場合のみ描画
            if conf >= threshold:
                color = cmap(i, bytes=True)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline=color, width=3)
                draw.text([bbox[0], bbox[1]-20], class_name, fill=color)

    return image