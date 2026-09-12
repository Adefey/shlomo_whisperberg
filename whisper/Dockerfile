FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

COPY . /app

WORKDIR /app

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg python3 python3-pip

RUN python3 -m pip install -r requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "4321", "router:app"]
