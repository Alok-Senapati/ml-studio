DEFAULT_GOAL := help

VERSION ?= 1.0.0
DOCKER ?= docker
COMPOSE ?= docker compose
ROOT_DIR := $(CURDIR)
COMPOSE_ENV_FILE := compose/.env

BASE_IMAGE := ai-base
PYTHON_IMAGE := ai-python
SCIENCE_IMAGE := ai-science
PYTORCH_IMAGE := ai-pytorch
TENSORFLOW_IMAGE := ai-tensorflow
JUPYTER_PYTORCH_IMAGE := ai-jupyter-pytorch
JUPYTER_TENSORFLOW_IMAGE := ai-jupyter-tensorflow
MLFLOW_IMAGE := ai-mlflow

BASE_TAG := $(BASE_IMAGE):$(VERSION)
PYTHON_TAG := $(PYTHON_IMAGE):$(VERSION)
SCIENCE_TAG := $(SCIENCE_IMAGE):$(VERSION)
PYTORCH_TAG := $(PYTORCH_IMAGE):$(VERSION)
TENSORFLOW_TAG := $(TENSORFLOW_IMAGE):$(VERSION)
JUPYTER_PYTORCH_TAG := $(JUPYTER_PYTORCH_IMAGE):$(VERSION)
JUPYTER_TENSORFLOW_TAG := $(JUPYTER_TENSORFLOW_IMAGE):$(VERSION)
MLFLOW_TAG := $(MLFLOW_IMAGE):$(VERSION)
PYTHON ?= python3
UV ?= uv

.PHONY: \
	help \
	build-all ci-build verify-all verify-ci \
	build-base run-base verify-base shell-base inspect-base history-base clean-base \
	build-python run-python verify-python shell-python \
	build-science run-science verify-science shell-science \
	build-pytorch run-pytorch verify-pytorch verify-pytorch-ci shell-pytorch \
	build-tensorflow run-tensorflow verify-tensorflow verify-tensorflow-ci shell-tensorflow \
	build-jupyter-pytorch build-jupyter-tensorflow verify-jupyter-pytorch verify-jupyter-tensorflow \
	build-mlflow verify-mlflow \
	up-pytorch down-pytorch logs-pytorch \
	up-tensorflow down-tensorflow logs-tensorflow \
	up-mlflow down-mlflow logs-mlflow

define docker_build
	$(DOCKER) build -t $(1) -f $(2) $(3)
endef

define docker_run
	$(DOCKER) run --rm -it $(1) $(2)
endef

define docker_verify
	$(DOCKER) run --rm $(1) \
		-v $(ROOT_DIR)/$(2):/workspace \
		-w /workspace \
		$(3) \
		bash verify.sh
endef

define docker_verify_ci
	$(DOCKER) run --rm \
		-e ml_studio_CI=1 \
		-v $(ROOT_DIR)/$(1):/workspace \
		-w /workspace \
		$(2) \
		bash verify.sh
endef

define compose_up
	$(COMPOSE) --env-file $(COMPOSE_ENV_FILE) -f $(1) up -d
endef

define compose_down
	$(COMPOSE) -f $(1) down
endef

define compose_logs
	$(COMPOSE) -f $(1) logs -f
endef

help:
	@echo "AI Workstation commands"
	@echo ""
	@echo "Build layers:"
	@echo "  make build-base build-python build-science build-pytorch build-tensorflow"
	@echo "  make build-jupyter-pytorch build-jupyter-tensorflow build-mlflow"
	@echo "  make ci-build"
	@echo ""
	@echo "Verify layers:"
	@echo "  make verify-all"
	@echo "  make verify-ci"
	@echo ""
	@echo "Run shells:"
	@echo "  make shell-base shell-python shell-science shell-pytorch shell-tensorflow"
	@echo ""
	@echo "Compose services:"
	@echo "  make up-pytorch down-pytorch logs-pytorch"
	@echo "  make up-tensorflow down-tensorflow logs-tensorflow"
	@echo "  make up-mlflow down-mlflow logs-mlflow"

# Base image
inspect-base:
	$(DOCKER) inspect $(BASE_TAG)

history-base:
	$(DOCKER) history $(BASE_TAG)

clean-base:
	$(DOCKER) image rm $(BASE_TAG)

build-base:
	$(call docker_build,$(BASE_TAG),docker/00-base/Dockerfile,docker/00-base)

run-base:
	$(call docker_run,,$(BASE_TAG))

verify-base:
	$(call docker_verify,,docker/00-base,$(BASE_TAG))

shell-base:
	$(call docker_run,,$(BASE_TAG))

# Python and science layers
build-python:
	$(call docker_build,$(PYTHON_TAG),docker/01-python/Dockerfile,docker/01-python)

run-python:
	$(call docker_run,,$(PYTHON_TAG))

verify-python:
	$(call docker_verify,,docker/01-python,$(PYTHON_TAG))

shell-python:
	$(call docker_run,,$(PYTHON_TAG))

build-science:
	$(call docker_build,$(SCIENCE_TAG),docker/02-science/Dockerfile,docker/02-science)

run-science:
	$(call docker_run,,$(SCIENCE_TAG))

verify-science:
	$(call docker_verify,,docker/02-science,$(SCIENCE_TAG))

shell-science:
	$(call docker_run,,$(SCIENCE_TAG))

# Framework layers
build-pytorch:
	$(call docker_build,$(PYTORCH_TAG),docker/03-pytorch/Dockerfile,docker/03-pytorch)

run-pytorch:
	$(call docker_run,--gpus all,$(PYTORCH_TAG))

verify-pytorch:
	$(call docker_verify,--gpus all,docker/03-pytorch,$(PYTORCH_TAG))

verify-pytorch-ci:
	$(call docker_verify_ci,docker/03-pytorch,$(PYTORCH_TAG))

shell-pytorch:
	$(call docker_run,--gpus all,$(PYTORCH_TAG))

build-tensorflow:
	$(call docker_build,$(TENSORFLOW_TAG),docker/03-tensorflow/Dockerfile,docker/03-tensorflow)

run-tensorflow:
	$(call docker_run,--gpus all,$(TENSORFLOW_TAG))

verify-tensorflow:
	$(call docker_verify,--gpus all,docker/03-tensorflow,$(TENSORFLOW_TAG))

verify-tensorflow-ci:
	$(call docker_verify_ci,docker/03-tensorflow,$(TENSORFLOW_TAG))

shell-tensorflow:
	$(call docker_run,--gpus all,$(TENSORFLOW_TAG))

build-all: build-base build-python build-science build-pytorch build-tensorflow

ci-build: build-all build-jupyter-pytorch build-jupyter-tensorflow build-mlflow

verify-all: verify-base verify-python verify-science verify-pytorch verify-tensorflow

verify-ci: verify-base verify-python verify-science verify-pytorch-ci verify-tensorflow-ci verify-jupyter-pytorch verify-jupyter-tensorflow verify-mlflow

# Jupyter images
build-jupyter-pytorch:
	$(DOCKER) build \
		--build-arg BASE_IMAGE=$(PYTORCH_TAG) \
		-t $(JUPYTER_PYTORCH_TAG) \
		-f docker/04-jupyter/Dockerfile .

build-jupyter-tensorflow:
	$(DOCKER) build \
		--build-arg BASE_IMAGE=$(TENSORFLOW_TAG) \
		-t $(JUPYTER_TENSORFLOW_TAG) \
		-f docker/04-jupyter/Dockerfile .

verify-jupyter-pytorch:
	$(call docker_verify,,docker/04-jupyter,$(JUPYTER_PYTORCH_TAG))

verify-jupyter-tensorflow:
	$(call docker_verify,,docker/04-jupyter,$(JUPYTER_TENSORFLOW_TAG))

# Compose-managed services
up-pytorch:
	$(call compose_up,compose/pytorch.yml)

down-pytorch:
	$(call compose_down,compose/pytorch.yml)

logs-pytorch:
	$(call compose_logs,compose/pytorch.yml)

up-tensorflow:
	$(call compose_up,compose/tensorflow.yml)

down-tensorflow:
	$(call compose_down,compose/tensorflow.yml)

logs-tensorflow:
	$(call compose_logs,compose/tensorflow.yml)

# MLflow image and service
build-mlflow:
	$(DOCKER) build -t $(MLFLOW_TAG) -f docker/05-mlflow/Dockerfile .

verify-mlflow:
	$(call docker_verify,,docker/05-mlflow,$(MLFLOW_TAG))

up-mlflow:
	$(call compose_up,compose/mlflow.yml)

down-mlflow:
	$(call compose_down,compose/mlflow.yml)

logs-mlflow:
	$(call compose_logs,compose/mlflow.yml)

new-project:
	$(UV) run python -m ml_studio.project.cli \
		--name $(NAME) \
		--description "$(DESCRIPTION)" \
		$(if $(SYNC),--sync,)