FROM public.ecr.aws/lambda/python:3.11

COPY pyproject.toml README.md ./
COPY core/ core/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . mangum

COPY services/inference/ .

CMD ["handler.handler"]
