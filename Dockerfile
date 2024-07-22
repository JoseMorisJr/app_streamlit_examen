
# Usa una imagen base de CUDA
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Actualiza e instala dependencias necesarias
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    curl \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Instala PyTorch y otras dependencias en una versión compatible con CUDA 11.0
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Instalar NVIDIA Container Toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && wget -qO - https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && wget -qO /etc/apt/sources.list.d/nvidia-docker.list https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit \
    && apt-get clean

# Configurar el runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip install --upgrade huggingface_hub

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo requirements.txt y el resto de tu aplicación al contenedor
COPY . .

# Instala las dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt


# Define el comando por defecto a ejecutar
CMD ["streamlit", "run", "app.py"]