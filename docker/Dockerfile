FROM python:3.6

RUN apt-get update && \
    apt-get install -y \
    libglu1-mesa

RUN apt-get update && \
    apt-get install -y \
    libvtk6-dev

ENV LIBGL_ALWAYS_INDIRECT 1

ADD ./requirements.txt /app/requirements.txt

# Install requirements
RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements.txt
    
# Install and enable mayavi notebook extension
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