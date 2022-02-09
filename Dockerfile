FROM rocker/rstudio

RUN apt update -y \
    && apt install -y --no-install-recommends default-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/