FROM almalinux:9

# Install dependencies for CMSSW (use --allowerasing for curl conflict)
RUN dnf -y update && dnf -y install --allowerasing \
    git \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    curl \
    ca-certificates \
    gcc \
    gcc-c++ \
    make \
    time \
    which \
    hostname \
    glibc-langpack-en \
    procps-ng \
    brotli \
    libbrotli \
    libnghttp2 \
    libidn2 \
    libpsl \
    libssh \
    openssl-libs \
    krb5-libs \
    zlib \
    && dnf clean all

# Set up locale
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Create non-root user
RUN useradd -m -s /bin/bash cmssw && \
    mkdir -p /work /outputs && \
    chown -R cmssw:cmssw /work /outputs

# Set working directory
WORKDIR /work

# Entry point will be provided at runtime
CMD ["/bin/bash"]
