FROM mambaorg/micromamba:latest

USER root
RUN apt update && apt -y install git

COPY --chown=$MAMBA_USER:$MAMBA_USER devtools/conda-envs/test_env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes