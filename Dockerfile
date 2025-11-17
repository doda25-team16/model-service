# Dockerfile
FROM python:3.12.9-slim
WORKDIR /root
COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY src/ /root/src/
COPY smsspamcollection/ /root/smsspamcollection/
COPY output/ /root/output/
ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]
