FROM python:3.6

LABEL maintainer="Syed Fahad Saeed"

RUN apt-get update && apt-get install python-opencv && apt-get update

WORKDIR /api

COPY . /api

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python","main.py"]

