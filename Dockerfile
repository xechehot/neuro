FROM continuumio/miniconda3:4.6.14

RUN mkdir /code

WORKDIR /code

COPY ./conda_environment.yml /code/conda_environment.yml

RUN conda env create -f /code/conda_environment.yml && \
    conda clean --all --yes

RUN adduser --disabled-password myuser
USER myuser
RUN conda init bash
RUN echo "conda activate nero-ml" >> ~/.bashrc

COPY . /app
WORKDIR /app

CMD ["/bin/bash", "-c", "source activate nero-ml && gunicorn --bind 0.0.0.0:$PORT run"]
