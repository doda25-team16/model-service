# Dockerfile

# stage 1: install dependencies
FROM python:3.12.9-slim AS builder

WORKDIR /build

COPY requirements.txt .
# (do not cache and use user install)
RUN pip install --no-cache-dir --user -r requirements.txt

# stage 2: final runtime image: install in root
FROM python:3.12.9-slim

WORKDIR /root

COPY --from=builder /root/.local /root/.local

COPY src/ /root/src/
COPY smsspamcollection/ /root/smsspamcollection/
COPY output/ /root/output/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/src
ENV MODEL_PORT="8081"

EXPOSE 8081

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]
