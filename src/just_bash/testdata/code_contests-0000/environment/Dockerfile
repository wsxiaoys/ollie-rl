FROM ubuntu:24.04

WORKDIR /app
RUN mkdir -p /output && chmod 777 /output

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
