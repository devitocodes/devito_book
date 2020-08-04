FROM python:3.6

RUN apt-get update && apt-get install -y -q \
    mpich \
    libmpich-dev \
    libgl1-mesa-glx \
    qt5-default \
    libx11-xcb1

ADD ./requirements.txt /app/requirements.txt

RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir jupyter && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements.txt

RUN /venv/bin/pip install RISE && \
    /venv/bin/jupyter nbextension install --py mayavi --user && \
    /venv/bin/jupyter nbextension enable mayavi --user --py
    
ADD ./fdm-devito-notebooks /app/fdm-devito-notebooks
COPY setup.cfg /app/

ADD docker/run-jupyter.sh /jupyter
ADD docker/entrypoint.sh /docker-entrypoint.sh

RUN chmod +x \
    /jupyter \
    /docker-entrypoint.sh

WORKDIR /app

ENV DEVITO_ARCH="gcc"
ENV DEVITO_LANGUAGE="openmp"

EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]


