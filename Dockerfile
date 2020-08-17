# Credit to https://github.com/FelipeLema/mne-binder/blob/master/Dockerfile

FROM jupyter/minimal-notebook:65761486d5d3

# Install core debian packages
USER root
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
    && apt-get install -yq --no-install-recommends \
    openssh-client \
    vim \
    curl \
    gcc \
    && apt-get clean

# Xvfb
RUN apt-get install -yq --no-install-recommends \
    xvfb \
    x11-utils \
    libx11-dev \
    qt5-default \
    && apt-get clean

ENV DISPLAY=:99

# Switch to notebook user
USER $NB_UID

ADD ./requirements.txt /app/requirements.txt

# Upgrade the package managers
RUN pip install --upgrade pip
RUN npm i npm@latest -g

# Install Python packages
RUN pip install vtk==8.1.2 && \
    pip install boto && \
    pip install h5py && \
    pip install nose && \
    pip install ipyevents && \
    pip install ipywidgets && \
    pip install mayavi && \
    pip install nibabel && \
    pip install numpy && \
    pip install pillow && \
    pip install pyqt5 && \
    pip install scikit-learn && \
    pip install scipy && \
    pip install xvfbwrapper && \
    pip install git+https://github.com/devitocodes/devito.git && \
    pip install https://github.com/nipy/PySurfer/archive/master.zip --ignore-installed certifi && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install Jupyter notebook extensions
RUN pip install RISE && \
    jupyter nbextension install rise --py --sys-prefix && \
    jupyter nbextension enable rise --py --sys-prefix && \
    jupyter nbextension install mayavi --py --sys-prefix && \
    jupyter nbextension enable mayavi --py --sys-prefix && \
    npm cache clean --force

# Try to decrease initial IPython kernel load times
RUN ipython -c "import matplotlib.pyplot as plt; print(plt)"

# Add notebooks
ADD ./fdm-devito-notebooks /app/fdm-devito-notebooks

RUN ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')

COPY setup.cfg /app/

WORKDIR /app

# Add an x-server to the entrypoint. This is needed by Mayavi
ENTRYPOINT ["tini", "-g", "--", "xvfb-run", "-a", "jupyter", "notebook"]
# CMD ["/jupyter"]