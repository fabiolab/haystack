FROM python:3.7.4-stretch

ENV HTTPS_PROXY=http://proxy.rd.francetelecom.fr:8080
ENV HTTP_PROXY=http://proxy.rd.francetelecom.fr:8080
#ENV HTTP_PROXY=http://10.193.21.110:8080
ENV https_proxy=http://proxy.rd.francetelecom.fr:8080
ENV http_proxy=http://proxy.rd.francetelecom.fr:8080
ENV NO_PROXY=registry.gitlab.tech.orange,registry.forge.orange-labs.fr,localhost,127.0.0.1

WORKDIR /home/user

RUN apt-get update && apt-get install -y \
    nano \
    curl  \
    git  \
    pkg-config  \
    cmake \
    libpoppler-cpp-dev  \
    tesseract-ocr  \
    libtesseract-dev  \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz && \
    tar -xvf xpdf-tools-linux-4.03.tar.gz && cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

# Copy Haystack code
COPY haystack /home/user/haystack/
# Copy package files & models
COPY setup.py setup.cfg pyproject.toml VERSION.txt LICENSE README.md models* /home/user/
# Copy REST API code
COPY rest_api /home/user/rest_api/

# Install package
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .[docstores,crawler,preprocessing,ocr,ray]
RUN pip install --no-cache-dir rest_api/
RUN ls /home/user
RUN pip freeze
RUN python3 -c "from haystack.utils.docker import cache_models;cache_models()"

# create folder for /file-upload API endpoint with write permissions, this might be adjusted depending on FILE_UPLOAD_PATH
RUN mkdir -p /home/user/rest_api/file-upload
RUN chmod 777 /home/user/rest_api/file-upload

# optional : copy sqlite db if needed for testing
#COPY qa.db /home/user/

# optional: copy data directory containing docs for ingestion
#COPY data /home/user/data
COPY data /home/user/data/
COPY feed-the-beast.py /home/user/
COPY feed-wikipedia.py /home/user/
COPY requester.py /home/user/

EXPOSE 8000

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app",  "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180"]
