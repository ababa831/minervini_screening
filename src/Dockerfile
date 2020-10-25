FROM ubuntu:18.04
MAINTAINER ababa831 "flvonlineconverter@gmail.com"

SHELL ["/bin/bash", "-c"]

WORKDIR /home

# 必要なpackageのインストール．
# editor関係は任意で追加可
RUN apt update --fix-missing
# Ref: https://qiita.com/yagince/items/deba267f789604643bab
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y  \
    tzdata \
    systemd \
    locales \
    iproute2 \
    curl \
    wget \
    git \
    sudo \
    unzip \ 
    python3-all-dev \
    python3-pip \
    python3-venv \
    nano
RUN rm -rf /root/.cache

# 地域，言語等の設定
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

# TimeZoneをJSTへ
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# Python関係の設定
RUN pip3 install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# 必要なパッケージのインストール
RUN pip install tqdm iexfinance python-dotenv yfinance yahoo_earnings_calendar xlrd schedule japanize-matplotlib slackclient nest_asyncio requests click

# 定義したエントリポイントをホストOSからコピー
COPY entrypoint.sh /home
COPY minervini_screening.py /home

ENTRYPOINT [ "./entrypoint.sh" ]