# test_predict.py

from predictor import predict_food

with open("pic.jpg", "rb") as f:
    image_bytes = f.read()

print(predict_food(image_bytes))