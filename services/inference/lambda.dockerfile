FROM public.ecr.aws/lambda/python:3.11

COPY core/ core/
COPY services/inference/requirements-lambda.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: \
      numpy==1.26.4 pandas scikit-learn onnxruntime onnx pyarrow && \
    pip install --no-cache-dir --only-binary=:all: -r requirements-lambda.txt

COPY services/inference/ .

CMD ["handler.handler"]
