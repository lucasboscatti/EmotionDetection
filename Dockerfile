FROM python:3.10-slim-buster

WORKDIR /python-docker

COPY requirements-emotion.txt requirements-emotion.txt
RUN pip3 install -r requirements-emotion.txt

COPY . .

CMD [ "python3", "app.py" ]