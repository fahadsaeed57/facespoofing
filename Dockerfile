FROM python:3.6

LABEL maintainer="Syed Fahad Saeed"

RUN apt-get update

WORKDIR /api

COPY . /api

RUN pip install -r requirements.txt

CMD ["bash","docker_start_script.sh"]

