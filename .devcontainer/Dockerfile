# Unified devcontainer for all simpleICP language implementations.
# Provides: C++ (g++/CMake/clangd/vcpkg with eigen3, cxxopts, nanoflann),
# Python 3, Octave, Julia, Rust, and MATLAB (via mpm).
#
# Note: linux/amd64 only. The MATLAB mpm binary is x86_64.
# On Apple Silicon hosts, run with --platform=linux/amd64.

ARG MATLAB_RELEASE=R2024b
FROM mathworks/matlab-deps:${MATLAB_RELEASE}

ARG DEBIAN_FRONTEND=noninteractive
ARG MATLAB_RELEASE
ARG JULIA_VERSION=1.10.4
ARG MATLAB_INSTALL_LOCATION=/opt/matlab

# --- Build tools, Python, Octave, X11 libs for MATLAB activation GUI ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo curl wget git ca-certificates xz-utils zip unzip locales gnupg \
        build-essential cmake ninja-build pkg-config gdb clangd \
        python3 python3-pip python3-venv \
        octave octave-statistics \
        libxt6 libxtst6 libxext6 libxrender1 libxi6 libxrandr2 \
        libgl1 libegl1 libxkbcommon0 libfontconfig1 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8

# --- vscode user with passwordless sudo ---
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# --- Python deps for the simpleicp package ---
# --break-system-packages is required because the matlab-deps base is Ubuntu
# 24.04 with PEP 668 enforced. We are in a single-purpose container, so
# installing system-wide is fine.
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy "scipy>=1.6" pandas lmfit matplotlib pytest twine

# --- Julia + simpleicp.jl deps in a shared depot ---
RUN JULIA_MAJOR=$(echo ${JULIA_VERSION} | cut -d. -f1,2) \
    && curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MAJOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
        | tar xz -C /opt/ \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia \
    && JULIA_DEPOT_PATH=/opt/julia-depot julia -e \
        'using Pkg; Pkg.add(["MultivariateStats", "NearestNeighbors", "StatsBase"]); Pkg.precompile()' \
    && chmod -R a+rwX /opt/julia-depot

ENV JULIA_DEPOT_PATH=/opt/julia-depot:/home/vscode/.julia

# --- Rust (system-wide, shared toolchain) ---
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path \
    && chmod -R a+rwX /usr/local/cargo /usr/local/rustup

# --- vcpkg + C++ deps (eigen3, cxxopts, nanoflann) ---
ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone --depth 1 https://github.com/microsoft/vcpkg ${VCPKG_ROOT} \
    && ${VCPKG_ROOT}/bootstrap-vcpkg.sh -disableMetrics \
    && ${VCPKG_ROOT}/vcpkg install eigen3 cxxopts nanoflann \
    && chown -R ${USERNAME}:${USERNAME} ${VCPKG_ROOT}

# --- MATLAB via mpm ---
RUN wget -q https://www.mathworks.com/mpm/glnxa64/mpm -O /tmp/mpm \
    && chmod +x /tmp/mpm \
    && /tmp/mpm install \
        --release=${MATLAB_RELEASE} \
        --destination=${MATLAB_INSTALL_LOCATION} \
        --products MATLAB Statistics_and_Machine_Learning_Toolbox \
    || (echo "mpm install failed:" && cat /tmp/mathworks_root.log && false) \
    && rm -f /tmp/mpm /tmp/mathworks_root.log \
    && ln -s ${MATLAB_INSTALL_LOCATION}/bin/matlab /usr/local/bin/matlab

USER ${USERNAME}
WORKDIR /workspaces/simpleICP
