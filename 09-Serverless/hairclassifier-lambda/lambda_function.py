#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import onnxruntime as ort
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor("xception", target_size=(200, 200))

session = ort.InferenceSession(
    "hair_classifier_v1.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
   "Wavy Hair",
   "Straight Hair",
   "Curly Hair",
   "Kinky/Coily Hair",
]


def predict(url):
    X = preprocessor.from_url(url)
    X = X.transpose(0, 3, 1, 2)  # Swaps axes: (Batch, H, W, C) -> (Batch, C, H, W)
    print(f"Shape sent to ONNX: {X.shape}")
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

