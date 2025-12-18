# ====================================================================
# STAGE 1: BUILDER - Used ONLY to extract the ONNX model files
# ====================================================================
#FROM agrigorev/model-2025-hairstyle:v1 as builder

# Note: We assume the ONNX files are at /app/ inside this image.
# If the build fails, we may need to find the correct path in the source image.

# ====================================================================
# STAGE 2: FINAL - The clean, optimized image for AWS Lambda
# ====================================================================
#WORKDIR /var/task
FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime keras-image-helper

COPY hair_classifier_v1.onnx.data .
COPY hair_classifier_v1.onnx .
COPY lambda_function.py .

# 3. Install required Python packages
# Must include onnxruntime and other preprocessing libraries
RUN pip install --upgrade pip
RUN pip install \
    onnxruntime \
    keras-image-helper \
    Pillow \
    numpy

CMD ["lambda_function.lambda_handler"]