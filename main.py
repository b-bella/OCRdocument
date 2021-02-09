import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
import uvicorn
import pytesseract
from pydantic import BaseModel
import io
import cv2
import requests

#pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = FastAPI()

APP_ROOT = os.path.dirname(__file__)

def read_img(img):
    text = pytesseract.image_to_string(img)
    return text

class ImageType(BaseModel):
    url: str

@app.get("/")
def main():
    return "OCR"

@app.post("/OCR")
async def OCRdocument(image_upload: UploadFile = File(...)):
    # file upload
    data = await image_upload.read()
    image_stream = io.BytesIO(data)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img_user = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # templete
    FILE_PATH = 'https://i.ibb.co/LhCrxqH/Query.png'
    response = requests.get(FILE_PATH)
    imageTemplete = io.BytesIO(response.content)
    imageTemplete.seek(0)
    file_bytes = np.asarray(bytearray(imageTemplete.read()), dtype=np.uint8)
    img_templete = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        per = 25
        pixelThreshold = 500

        roi = [[(98, 984), (680, 1074), 'text', 'Name'],
               [(740, 980), (1320, 1078), 'text', 'Phone'],
               [(100, 1418), (686, 1518), 'text', 'Email'],
               [(740, 1416), (1318, 1512), 'text', 'ID'],
               [(110, 1598), (676, 1680), 'text', 'City'],
               [(748, 1592), (1328, 1686), 'text', 'Country']]

        imgQ = img_templete
        h, w, c = imgQ.shape
        orb = cv2.ORB_create(10000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)

        img = img_user
        kp2, des2 = orb.detectAndCompute(img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]

        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        myData = dict()

        for x, r in enumerate(roi):
            cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

            if r[2] == 'text':
                x = read_img(imgCrop)
                myData[r[3]] = x

        return myData

    except Exception as e:
        return e



if __name__ == '__main__':
    uvicorn.run(app, debug=True)