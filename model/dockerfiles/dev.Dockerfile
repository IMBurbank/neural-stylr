ARG USE_GPU=
ARG _TAG_SUFFIX=${USE_GPU:+-gpu}
ARG TAG=1804${_TAG_SUFFIX}

FROM dget/dock-tf:${TAG}

COPY requirements.txt /

RUN pip install --no-cache-dir -r /requirements.txt

RUN $([ -z "$(which python)" ] && \
        echo "alias python=python3" >> /etc/bash.bashrc) \
    && mkdir /.local \
    && chmod a+rwx /.local

ENTRYPOINT ["/entry.sh"]
CMD ["bash", "-l"]