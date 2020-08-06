FROM python:3.6

USER root

RUN apt-get update && \
    apt-get install -y \
    libglu1-mesa

RUN apt-get update && \
    apt-get install -y \
    libvtk6-dev

# RUN apt-get update && \
#     apt-get install -y \
#     mayavi2

# RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# USER jovyan

ENV LIBGL_ALWAYS_INDIRECT 1

RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir vtk==8.1.2 && \
    /venv/bin/pip install --no-cache-dir mayavi && \
    /venv/bin/pip install --no-cache-dir jupyter && \
    /venv/bin/pip install --no-cache-dir jupyter_contrib_nbextensions
    
RUN python3 -m venv /venv && \
    /venv/bin/jupyter nbextension install --py mayavi --user && \
    /venv/bin/jupyter nbextension enable mayavi --user --py