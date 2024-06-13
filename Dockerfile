FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1 insightface==0.7.3 onnxruntime==1.18.0 onnxruntime-gpu==1.18.0 \
	ultralytics==8.2.27 segment-anything==1.0 piexif==1.1.3 qrcode==7.4.2 requirements-parser==0.9.0 rembg==2.0.57 rich==13.7.1 rich-argparse==1.5.1 matplotlib==3.8.4 pillow==10.3.0 spandrel==0.3.4 && \
    git clone https://github.com/camenduru/ComfyUI /content/ComfyUI && \
	git clone https://github.com/camenduru/comfy_mtb /content/ComfyUI/custom_nodes/comfy_mtb && \
	git clone https://github.com/camenduru/ComfyUI_IPAdapter_plus /content/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus && \
	git clone https://github.com/camenduru/ComfyUI-AnimateDiff-Evolved /content/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved && \
	git clone https://github.com/camenduru/ComfyUI-Image-Selector /content/ComfyUI/custom_nodes/ComfyUI-Image-Selector && \
	git clone https://github.com/camenduru/ComfyUI-Frame-Interpolation /content/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
	git clone https://github.com/camenduru/ComfyUI-VideoHelperSuite /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/563988 -d /content/ComfyUI/models/checkpoints -o zavychromaxl_v80.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors -d /content/ComfyUI/models/loras/SDXL-Lightning -o sdxl_lightning_8step_lora.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/KamCastle/jugg/resolve/main/juggernaut_reborn.safetensors -d /content/ComfyUI/models/checkpoints -o juggernaut_reborn.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors -d /content/ComfyUI/models/loras -o AnimateLCM_sd15_t2v_lora.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d /content/ComfyUI/models/vae -o vae-ft-mse-840000-ema-pruned.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt -d /content/ComfyUI/models/animatediff_models -o AnimateLCM_sd15_t2v.ckpt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth -d /content/ComfyUI/models/upscale_models -o RealESRGAN_x4.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -d /content/ComfyUI/models/clip_vision -o CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors -d /content/ComfyUI/models/ipadapter -o ip-adapter-plus-face_sd15.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors -d /content/ComfyUI/models/ipadapter -o ip-adapter-plus_sd15.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py
