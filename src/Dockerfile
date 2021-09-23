# python version 3.7
FROM python:3.7

# set workdir to src folder
WORKDIR /src

# copy the dependencies file to the working directory
COPY requirements.txt .

# install requirements
RUN pip3 install -r requirements.txt

# install spacy
RUN pip3 install -U pip setuptools wheel
RUN pip3 install -U spacy

# install spacy models
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy download en_core_web_lg

# start the jupyter notebook server on the 8888 port
EXPOSE 8888
CMD ["jupyter notebook --port=8888"]

