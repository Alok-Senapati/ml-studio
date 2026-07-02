inspect-base:
	docker inspect ai-base:1.0.0

history-base:
	docker history ai-base:1.0.0

clean-base:
	docker image rm ai-base:1.0.0

build-base:
	docker build -t ai-base:1.0.0 \
		-f docker/00-base/Dockerfile \
		docker/00-base

run-base:
	docker run --rm -it ai-base:1.0.0

verify-base:
	docker run --rm \
		-v $(PWD)/docker/00-base:/workspace \
		-w /workspace \
		ai-base:1.0.0 \
		bash verify.sh