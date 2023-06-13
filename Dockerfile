FROM python:3.10.7-alpine

EXPOSE 8501
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /opt/app

CMD streamlit run ./main.py
