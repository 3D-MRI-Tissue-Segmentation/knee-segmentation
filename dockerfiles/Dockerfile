FROM tf_aug:latest

WORKDIR /home/joe/github/knee-segmentation/

# COPY ../reqs.txt reqs.txt

# Install tensorflow and reqs
RUN pip install -U pip
RUN pip install -U tensorflow
RUN pip install -U tensorflow-addons
RUN pip install sklearn
RUN pip install pandas
RUN pip install pytest
RUN pip install matplotlib
RUN pip install flake8
RUN pip install jupyterlab

RUN apt-get update
RUN apt-get install -y tmux
RUN apt-get install -y nano
RUN apt-get install -y sudo

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
