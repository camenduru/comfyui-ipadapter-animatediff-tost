import torch
import random
import ast
from comfy.sd import load_checkpoint_guess_config
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file
import nodes
from comfy_extras import nodes_upscale_model
import numpy as np
from PIL import Image

import asyncio
import execution
import server
from nodes import load_custom_node

import os, json, requests, runpod

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server_instance)

load_custom_node("/content/ComfyUI/custom_nodes/comfy_mtb")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Image-Selector")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")

from nodes import NODE_CLASS_MAPPINGS
ColorCorrect = NODE_CLASS_MAPPINGS["Color Correct (mtb)"]()
IPAdapterAdvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
IPAdapterUnifiedLoader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()

ADE_LoadAnimateDiffModel = NODE_CLASS_MAPPINGS["ADE_LoadAnimateDiffModel"]()
ADE_MultivalDynamic = NODE_CLASS_MAPPINGS["ADE_MultivalDynamic"]()
ADE_ApplyAnimateDiffModel = NODE_CLASS_MAPPINGS["ADE_ApplyAnimateDiffModel"]()
ADE_LoopedUniformContextOptions = NODE_CLASS_MAPPINGS["ADE_LoopedUniformContextOptions"]()
ADE_AnimateDiffSamplingSettings = NODE_CLASS_MAPPINGS["ADE_AnimateDiffSamplingSettings"]()
ADE_UseEvolvedSampling = NODE_CLASS_MAPPINGS["ADE_UseEvolvedSampling"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
ImageScaleBy = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
ImageScale = NODE_CLASS_MAPPINGS["ImageScale"]()
UpscaleModelLoader = nodes_upscale_model.NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
ImageUpscaleWithModel = nodes_upscale_model.NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()

ImageSelector = NODE_CLASS_MAPPINGS["ImageSelector"]()
ImageBatch = NODE_CLASS_MAPPINGS["ImageBatch"]()
RIFE_VFI =  NODE_CLASS_MAPPINGS["RIFE VFI"]()

def set_last_layer(clip, stop_at_clip_layer):
    clip = clip.clone()
    clip.clip_layer(stop_at_clip_layer)
    return clip

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

with torch.inference_mode():
    model_patcher_1, clip_1, vae_1, clipvision_1 = load_checkpoint_guess_config("/content/ComfyUI/models/checkpoints/zavychromaxl_v80.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
    clip_1 = set_last_layer(clip_1, -1)
    lora_1 = load_torch_file("/content/ComfyUI/models/loras/SDXL-Lightning/sdxl_lightning_8step_lora.safetensors", safe_load=True)
    model_lora_1, clip_lora_1 = load_lora_for_models(model_patcher_1, clip_1, lora_1, 1.0, 0.0)

    model_patcher_2, clip_2, vae_2, clipvision_2 = load_checkpoint_guess_config("/content/ComfyUI/models/checkpoints/juggernaut_reborn.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
    clip_2 = set_last_layer(clip_2, -1)
    lora_2 = load_torch_file("/content/ComfyUI/models/loras/AnimateLCM_sd15_t2v_lora.safetensors", safe_load=True)
    model_lora_2, clip_lora_2 = load_lora_for_models(model_patcher_2, clip_2, lora_2, 1.0, 0.0)
    IPAdapterPlus_model = IPAdapterUnifiedLoader.load_models(model_lora_2, 'PLUS (high strength)', lora_strength=1.0, provider="CUDA", ipadapter=None)
        
    animate_vae = VAELoader.load_vae("vae-ft-mse-840000-ema-pruned.safetensors")

@torch.inference_mode()
def generate(input):
    values = input["input"]

    prompt_1 = values['prompt']
    negative_prompt_1 = values['negative_prompt']
    seed = values['seed']
    is_upscale = values['is_upscale']

    latent_zero_1 = {"samples":torch.zeros([1, 4, 1024 // 8, 1024 // 8])}
    cond_1, pooled_1 = clip_lora_1.encode_from_tokens(clip_lora_1.tokenize(prompt_1), return_pooled=True)
    cond_1 = [[cond_1, {"pooled_output": pooled_1}]]
    n_cond_1, n_pooled_1 = clip_lora_1.encode_from_tokens(clip_lora_1.tokenize(negative_prompt_1), return_pooled=True)
    n_cond_1 = [[n_cond_1, {"pooled_output": n_pooled_1}]]    
    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    print(seed)
    sample_1 = nodes.common_ksampler(model=model_lora_1, 
                            seed=seed, 
                            steps=8, 
                            cfg=1.0, 
                            sampler_name="euler", 
                            scheduler="sgm_uniform", 
                            positive=cond_1, 
                            negative=n_cond_1,
                            latent=latent_zero_1, 
                            denoise=1)
    sample_1 = sample_1[0]["samples"].to(torch.float16)
    vae_1.first_stage_model.cuda()
    decoded_1 = vae_1.decode_tiled(sample_1).detach()
    decoded_image_1 = ColorCorrect.correct(decoded_1, clamp=True, gamma=1.1, contrast=1.1, exposure=0.15, offset=0.0, hue=0.0, saturation=1.0, value=1.0)

    ip_model_patcher = IPAdapterAdvanced.apply_ipadapter(IPAdapterPlus_model[0], IPAdapterPlus_model[1], start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, image=decoded_image_1[0])
    motion_model = ADE_LoadAnimateDiffModel.load_motion_model(model_name="AnimateLCM_sd15_t2v.ckpt")
    scale_multival = ADE_MultivalDynamic.create_multival(float_val=1.2)
    m_models = ADE_ApplyAnimateDiffModel.apply_motion_model(motion_model=motion_model[0], scale_multival=scale_multival[0])
    context_options = ADE_LoopedUniformContextOptions.create_options(context_length=16, context_stride=1, context_overlap=4, closed_loop=False, fuse_method="flat", use_on_equal_length=False, start_percent=0.0, guarantee_steps=1)
    sample_settings = ADE_AnimateDiffSamplingSettings.create_settings(batch_offset=0, noise_type="FreeNoise", seed_gen="comfy", seed_offset=0)
    animate_diff_model = ADE_UseEvolvedSampling.use_evolved_sampling(model=ip_model_patcher[0], beta_schedule="sqrt_linear (AnimateDiff)", m_models=m_models[0], context_options=context_options[0], sample_settings=sample_settings[0])

    latent_zero_24 = EmptyLatentImage.generate(288, 512, 24)
    prompt_2= ""
    cond_2, pooled_2 = clip_lora_2.encode_from_tokens(clip_lora_2.tokenize(prompt_2), return_pooled=True)
    cond_2 = [[cond_2, {"pooled_output": pooled_2}]]
    negative_prompt_2 = ""
    n_cond_2, n_pooled_2 = clip_lora_2.encode_from_tokens(clip_lora_2.tokenize(negative_prompt_2), return_pooled=True)
    n_cond_2 = [[n_cond_2, {"pooled_output": n_pooled_2}]]
    sample_2 = nodes.common_ksampler(model=animate_diff_model[0], 
                            seed=seed, 
                            steps=9, 
                            cfg=1.0, 
                            sampler_name="lcm", 
                            scheduler="sgm_uniform", 
                            positive=cond_2, 
                            negative=n_cond_2,
                            latent=latent_zero_24[0], 
                            denoise=1)
    sample_2 = sample_2[0]["samples"].to(torch.float16)
    animate_vae[0].first_stage_model.cuda()
    decoded_video_1 = animate_vae[0].decode_tiled(sample_2).detach()

    upscaled_video_1 = ImageScaleBy.upscale(decoded_video_1, "lanczos", 2.0)
    latent_video_2 = VAEEncode.encode(animate_vae[0], upscaled_video_1[0])

    sample_3 = nodes.common_ksampler(model=animate_diff_model[0], 
                            seed=seed, 
                            steps=10, 
                            cfg=1.0, 
                            sampler_name="lcm", 
                            scheduler="sgm_uniform", 
                            positive=cond_2, 
                            negative=n_cond_2,
                            latent=latent_video_2[0], 
                            denoise=0.45)
    sample_3 = sample_3[0]["samples"].to(torch.float16)
    animate_vae[0].first_stage_model.cuda()
    decoded_video_2 = animate_vae[0].decode_tiled(sample_3).detach()

    if(is_upscale):
        upscale_model = UpscaleModelLoader.load_model("RealESRGAN_x4.pth")
        upscale_model_with_model = ImageUpscaleWithModel.upscale(upscale_model=upscale_model[0], image=decoded_video_2)
        decoded_video_2 = ImageScale.upscale(image=upscale_model_with_model[0], upscale_method="nearest-exact", width=1080, height=1920, crop="disabled")[0]

    image_selector = ImageSelector.run(images=decoded_video_2, selected_indexes="0")
    image_batch = ImageBatch.batch(image1=image_selector[0], image2=decoded_video_2)
    rife_vfi = RIFE_VFI.vfi(frames=image_batch[0], ckpt_name="rife47.pth", clear_cache_after_n_frames=320, multiplier=4, fast_mode=True, ensemble=True, scale_factor = 1.0)

    prompt_3 = "[{'inputs': {'pix_fmt': 'yuv420p', 'crf': 15, 'save_metadata': True}}]"
    prompt_3 = ast.literal_eval(prompt_3)
    out_video4 = VHS_VideoCombine.combine_video(images=rife_vfi[0], frame_rate=30, loop_count=0, filename_prefix="interpolated/", format="video/h264-mp4", save_output=False, prompt=prompt_3, unique_id=0)

    result = out_video4["result"][0][1][1]
    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})