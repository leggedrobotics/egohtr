FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the actual Environment Variable based on the ARG
ARG HOME_ARG="/root"
ENV HOME=${HOME_ARG} 
ENV HGT_VENV_DIR=${HOME}/hgt
ENV VGGT_VENV_DIR=${HOME}/vggt
ENV PROJECT_ROOT=${HOME}/human_terrain_generation


# Install basics
RUN --mount=type=cache,id=core_apt_cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=core_apt_cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y \
    software-properties-common \
    curl wget git build-essential \
    libgl1 \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*


### vggt venv
WORKDIR ${HOME}
COPY requirements_vggt.txt ${HOME}/

RUN --mount=type=cache,target=${HOME}/.cache/pip \
    python3.12 -m venv ${VGGT_VENV_DIR} \
    && ${VGGT_VENV_DIR}/bin/pip install --upgrade pip setuptools wheel \
    && ${VGGT_VENV_DIR}/bin/pip install -r requirements_vggt.txt \
    && rm -f requirements_vggt.txt

RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${VGGT_VENV_DIR}/bin/pip install \
    "torch>=2.10.0.dev" \
    "torchaudio>=2.8.0.dev" \
    "torchvision>=0.25.0.dev" \
    --index-url https://download.pytorch.org/whl/nightly/cu128

### HGT venv
WORKDIR ${HOME}
COPY requirements.txt ${HOME}/

# RUN python3.11 -m venv ${HGT_VENV_DIR} \
# && source ${HGT_VENV_DIR}/bin/activate

RUN --mount=type=cache,target=${HOME}/.cache/pip \
    python3.11 -m venv ${HGT_VENV_DIR} \
    && ${HGT_VENV_DIR}/bin/pip install --upgrade pip setuptools wheel


RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${HGT_VENV_DIR}/bin/pip install \
    torch==2.9.0 \
    torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${HGT_VENV_DIR}/bin/pip install -r requirements.txt \
    && rm requirements.txt \
    && echo "source ${HGT_VENV_DIR}/bin/activate" >> ${HOME}/.bashrc


RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${HGT_VENV_DIR}/bin/pip install \
    --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

WORKDIR ${PROJECT_ROOT}
RUN mkdir -p ${PROJECT_ROOT}/external
COPY ../external/GMR ${PROJECT_ROOT}/external/GMR
RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${HGT_VENV_DIR}/bin/pip install \
    -e ${PROJECT_ROOT}/external/GMR

