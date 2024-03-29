# Ubuntu Linux as the base image. You can use any version of Ubuntu here
FROM ubuntu:18.04
# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# Install Python
RUN apt-get -y update && \
apt-get -y upgrade
# The following line ensures that the subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip python3-dev
# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QA
ADD . /QA
COPY requirements.txt /QA/requirements.txt
# Change the permissions of programs
CMD ["chmod 777 /QA/*"]
# Set working dir as /QA
WORKDIR /QA
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r /QA/requirements.txt
ENTRYPOINT ["/bin/bash", "-c"]
