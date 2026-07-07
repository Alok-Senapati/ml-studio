IMAGE_BASE=ai-base
BASE_VERSION=1.0.0

IMAGE_PYTHON=ai-python
PYTHON_VERSION=1.0.0

inspect-base:
	docker inspect $(IMAGE_BASE):$(BASE_VERSION)

history-base:
	docker history $(IMAGE_BASE):$(BASE_VERSION)

clean-base:
	docker image rm $(IMAGE_BASE):$(BASE_VERSION)

build-base:
	docker build -t $(IMAGE_BASE):$(BASE_VERSION) \
		-f docker/00-base/Dockerfile \
		docker/00-base

run-base:
	docker run --rm -it $(IMAGE_BASE):$(BASE_VERSION)

verify-base:
	docker run --rm \
		-v $(PWD)/docker/00-base:/workspace \
		-w /workspace \
		$(IMAGE_BASE):$(BASE_VERSION) \
		bash verify.sh

build-python:
	docker build \
	-t $(IMAGE_PYTHON):$(PYTHON_VERSION) \
	-f docker/01-python/Dockerfile \
	docker/01-python

run-python:
	docker run --rm -it \
	$(IMAGE_PYTHON):$(PYTHON_VERSION)

verify-python:
	docker run --rm \
	-v $(PWD)/docker/01-python:/workspace \
	-w /workspace \
	$(IMAGE_PYTHON):$(PYTHON_VERSION) \
	bash verify.sh