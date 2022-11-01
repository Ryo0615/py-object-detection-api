# 物体検出 API の構築 🧑‍💻

受け取った画像に対して、物体検出を行う API を構築してください。

## はじめに

1. 本リポジトリをフォークしてください
2. フォークしたリポジトリをローカルにクローンして開発を行ってください

## 要件

- Python のバージョンや使用するライブラリは問いません 🐍
- 物体検出に用いるモデルやフレームワークは問いません
- COCO データセットに含まれる「車」「人」「犬」「自転車」などのクラスを検出できることを想定しています
  - 一からモデルを作成する必要は必ずしもなく、事前学習済みモデル等を有効に活用してください
- Dockerfile を作成し、デプロイ可能な状態を作ってください 🐳

## API 仕様

- `Content-Type: multipart/form-data` で送信された画像に対して物体検出を行い、その結果を画像として返す
- クエリパラメータに `threshold` を指定すると、その値以上の confidence の物体のみを返す（デフォルトは 25%）

### 入出力イメージ

Bounding box の色や文字の位置などはこれに限りません

#### Input

![input](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/dog.jpg)

#### Output

![output](https://miro.medium.com/max/768/1*DqEFJex8max3s9jBkK2_-g.png)

## 私たちが見させていただくポイント 👀

- API に関する基礎知識
- Python に関する基礎知識
- 機械学習ライブラリの使用方法
- git の使い方
- Docker 周りの知識
