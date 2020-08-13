# Credit goes to https://taku-y.github.io/mac-docker-jupyter-mayavi.html

FROM python:3.6

USER root

RUN apt-get update && \
    apt-get install -y \
    libglu1-mesa

RUN apt-get update && \
    apt-get install -y \
    libvtk6-dev

RUN apt-get update && \
    apt-get -y install libgl1-mesa-glx libgl1-mesa-dri && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y xvfb

# RUN apt-get update && \
#     apt-get install -y pkg-config

# # Install nvidia driver
# RUN apt-get purge nvidia* && \
#     apt-get autoremove && \
#     dpkg -P cuda-repo-ubuntu1404

# RUN cd ~ && \
#     wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.69/NVIDIA-Linux-x86_64-384.69.run

# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     apt-get install -y gcc-multilib && \
#     apt-get install -y dkms

# RUN cd ~ && \
#     chmod +x NVIDIA-Linux-x86_64-384.69.run && \
#     ./NVIDIA-Linux-x86_64-384.69.run --dkms -s --no-opengl-files

ENV LIBGL_ALWAYS_INDIRECT 1

ADD ./requirements.txt /app/requirements.txt

RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements.txt
    
RUN python3 -m venv /venv && \
    /venv/bin/jupyter nbextension install --py mayavi --user && \
    /venv/bin/jupyter nbextension enable mayavi --user --py

# Add notebooks
ADD ./fdm-devito-notebooks /app/fdm-devito-notebooks
COPY setup.cfg /app/

ADD docker/run-jupyter.sh /jupyter
ADD docker/entrypoint.sh /docker-entrypoint.sh

RUN chmod +x \
    /jupyter \
    /docker-entrypoint.sh

WORKDIR /app

# Run notebooks on port 5000
EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]