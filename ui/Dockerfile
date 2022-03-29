FROM python:3.7.4-stretch

ENV HTTPS_PROXY=http://proxy.rd.francetelecom.fr:8080
ENV HTTP_PROXY=http://proxy.rd.francetelecom.fr:8080
#ENV HTTP_PROXY=http://10.193.21.110:8080
ENV https_proxy=http://proxy.rd.francetelecom.fr:8080
ENV http_proxy=http://proxy.rd.francetelecom.fr:8080
ENV NO_PROXY=registry.gitlab.tech.orange,registry.forge.orange-labs.fr,localhost,127.0.0.1,.rd.francetelecom.fr,yd-deskin-demo.rd.francetelecom.fr

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# copy code
RUN mkdir ui/
COPY setup.py /home/user/ui

# install as a package
RUN pip install --upgrade pip
RUN pip install ui/

COPY utils.py /home/user/ui
COPY webapp.py /home/user/ui
COPY webapp_arod.py /home/user/ui
COPY eval_labels_example.csv /home/user/


EXPOSE 8501

# cmd for running the API
CMD ["python", "-m", "streamlit", "run", "ui/webapp.py"]
