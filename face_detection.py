#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# パソコンの内臓カメラとOpenCV (3.1.0) を使って顔認識に挑戦
#
# ファイルに実行権限を付加してから ./face_detection.py と打つと動く。
# 止めるときにはESCキーを押す。
#
# OpenCV (3.1.0) のインストールにはAnaconda (Python 2.7用) がおすすめ。
# 以下のワンラインコマンドでOpenCV 3.1.0 が入る。
# conda install -c https://conda.binstar.org/menpo opencv3
#
# Author: Makoto Otsuka

import cv2    # OpenCV 3.1.0 をインポート
import os
import numpy as np

def detect_face(mirror=True, size=None):
    # 内臓カメラからビデオ画像をキャプチャして顔認識する

    # カメラをキャプチャする
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    # カスケード分類器で用いる顔認識用のXMLファイルを読み込む
    cascade_path = os.path.join(os.environ['HOME'],     "anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    # 検出された場所を囲むための矩形の色
    color = (240, 240, 240) # 灰色

    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()
        if ret == False:
            break

        # 鏡のように映るか否か
        if mirror is True:
            frame = cv2.flip(frame, 1)    # ミラーイメージを取得

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # なぜかこれが必要... 謎...
        frame = frame.copy()

        # グレースケールに変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 画像の中にある顔を認識
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

        # 検出した顔の位置座標をすべて表示
        #print("{0}\n".format(facerect))

        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=7)

        # フレームを表示する
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    detect_face(mirror=True, size=None)
