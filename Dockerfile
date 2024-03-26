FROM ubuntu:latest
LABEL authors="anagle"

ENTRYPOINT ["top", "-b"]