.PHONY: build run bash stop 

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

IMAGE_NAME ?= auxiliary_modular_addition
CONTAINER_NAME ?= auxiliary_modular_addition-container
DOCKER_GPUS ?= all

build:
	docker build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t $(IMAGE_NAME) .

run:
	docker run --gpus $(DOCKER_GPUS) -d --name $(CONTAINER_NAME) -v "$(CURDIR)":/app $(IMAGE_NAME) tail -f /dev/null

bash:
	docker exec -it $(CONTAINER_NAME) /bin/bash

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
