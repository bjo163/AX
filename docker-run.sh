#!/bin/bash

# Nama image yang akan dibuat
IMAGE_NAME="ax-app"

# Nama container yang akan dijalankan
CONTAINER_NAME="ax-app-container"

# Perintah untuk build image
docker build -t $IMAGE_NAME .

# Perintah untuk menjalankan container
docker rm -f $CONTAINER_NAME
docker run -d -p 8001:8001/tcp --name $CONTAINER_NAME $IMAGE_NAME