FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 as base 

# Install common dependencies
RUN apt-get update -qqy && \
    apt-get install -y \
        locales \
        software-properties-common \
        ffmpeg \
        libsm6 \
        libxext6 

# Install python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3-pip && \
    apt-get clean && \
    apt-get autoremove
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python -m pip install -U pip setuptools wheel && \
    python -m pip install --upgrade pip 

# locale setting & env(s)
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

# install requirements
WORKDIR /opt/ml/app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt --no-cache-dir 

# copy source
COPY . $WORKDIR

EXPOSE 5555
RUN mkdir -p log 

# Entrypoint 
CMD ["streamlit", "run", "app.py", "--server.port=5555"]
