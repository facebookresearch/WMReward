# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5TokenizerFast
from PIL import Image

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan, CosmosTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_cosmos_guardrail_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.cosmos.pipeline_output import CosmosPipelineOutput
from diffusers.utils import export_to_video

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# Optional GPU stats (best-effort import)
try:
    import gpustat  # type: ignore
except Exception:
    gpustat = None

import sys
from utils import generate_vjepa_masks, apply_masks

if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import Cosmos2VideoToWorldPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> # Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
        >>> model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
        >>> pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
        >>> negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
        ... )

        >>> video = pipe(
        ...     image=image, prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
        ... ).frames[0]
        >>> export_to_video(video, "output.mp4", fps=16)
        ```
"""

def print_gpu_memory(if_vis, info=""):
    if not if_vis or gpustat is None:
        return
    query = gpustat.new_query()
    for idx in range(1):
        gpu = query.gpus[idx]
        print(f"{info}")
        print(f"GPU ID: {gpu.index}")
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Utilization: {gpu.utilization}%")
        print(f"Memory Used: {gpu.memory_used} MB / {gpu.memory_total} MB")
        print(f"Temperature: {gpu.temperature}°C")
        print("-" * 20)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Cosmos2VideoToWorldPipeline(DiffusionPipeline):
    r"""
    Pipeline for video-to-world generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. Cosmos uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CosmosTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # We mark safety_checker as optional here to get around some test failures, but it is not really optional
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        safety_checker: CosmosSafetyChecker = None,
    ):
        super().__init__()

        if safety_checker is None:
            # safety_checker = CosmosSafetyChecker()
            safety_checker = None

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        if self.scheduler is not None:
            self.scheduler.register_to_config(
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                sigma_data=self.sigma_data,
                final_sigmas_type=self.final_sigmas_type,
            )


        # self.vjepa_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vith-fpc64-256")
        # self.vjepa_model = AutoModel.from_pretrained(
        #     "facebook/vjepa2-vith-fpc64-256",
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     attn_implementation="sdpa"
        # )

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_length=True,
            return_offsets_mapping=False,
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=prompt_attention_mask
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        lengths = prompt_attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            prompt_embeds[i, length:] = 0

        return prompt_embeds

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        video: torch.Tensor,
        batch_size: int,
        num_channels_latents: 16,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        do_classifier_free_guidance: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_cond_frames = video.size(2)
        if num_cond_frames >= num_frames:
            # Take the last `num_frames` frames for conditioning
            num_cond_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            video = video[:, :, -num_frames:]
        else:
            num_cond_latent_frames = (num_cond_frames - 1) // self.vae_scale_factor_temporal + 1
            num_padding_frames = num_frames - num_cond_frames
            last_frame = video[:, :, -1:]
            padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
            video = torch.cat([video, padding], dim=2)

        if isinstance(generator, list):
            init_latents = [
                retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator=generator[i])
                for i in range(batch_size)
            ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        init_latents = (init_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.config.sigma_max

        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        uncond_indicator = uncond_mask = None
        if do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding

        return latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sigma_conditioning: float = 0.0001,
        guidance_step: Union[List[int], int] = 0,
        guidance_lr: Union[float, List[float]] = 1e-2,
        guidance_frequency: int = 1,
        loss_fn: str = "slice_pred",
        additional_inputs: Optional[Dict[str, Any]] = None,
        travel_time: Tuple[int, int] = (0, 50),
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                The image to be used as a conditioning input for the video generation.
            video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                The video to be used as a conditioning input for the video generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            fps (`int`, defaults to `16`):
                The frames per second of the generated video.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.
            sigma_conditioning (`float`, defaults to `0.0001`):
                The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                set to a small value close to zero.

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # if self.safety_checker is None:
        #     raise ValueError(
        #         f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
        #         "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
        #         f"Please ensure that you are compliant with the license agreement."
        #     )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        # Set requires_grad_(False) for all models to save memory
        self.text_encoder.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.transformer.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.vae.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad = False

        # Keep V-JEPA model gradients disabled for guidance
        # self.vjepa_model.requires_grad_(False)
        # for p in self.vjepa_model.parameters():
        #     p.requires_grad = False

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # Disable safety checker for this pipeline variant
        self.safety_checker = None
        # torch.cuda.empty_cache()

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # print_gpu_memory(True, "After encode_prompt")

        # 4. Prepare timesteps
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            # Replace the last sigma (which is zero) with the minimum sigma value
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        # 5. Prepare latent variables
        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        self.transformer.enable_gradient_checkpointing()
        # self.vae.enable_gradient_checkpointing()

        if image is not None:
            video = self.video_processor.preprocess(image, height, width).unsqueeze(2)
        else:
            video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=vae_dtype)

        with torch.no_grad():

            num_channels_latents = self.transformer.config.in_channels - 1
            latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = self.prepare_latents(
                video,
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                self.do_classifier_free_guidance,
                torch.float32,
                device,
                generator,
                latents,
            )
            unconditioning_latents = None

        cond_mask = cond_mask.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # print_gpu_memory(True, "After prepare_latents")

        # VJEPA setup for guidance
        if isinstance(guidance_step, int):
            guidance_step = [guidance_step] * num_inference_steps
        else:
            assert len(guidance_step) == num_inference_steps, "guidance_step must be a list of length num_inference_steps"

        if isinstance(guidance_lr, float):
            guidance_lr = [guidance_lr] * num_inference_steps
        else:
            assert len(guidance_lr) == num_inference_steps, "guidance_lr must be a list of length num_inference_steps"

        # Match CogVideoX schedule semantics
        shorten_steps = 35 - num_inference_steps
        guidance_lr = guidance_lr[-num_inference_steps:]
        guidance_step = guidance_step[-num_inference_steps:]
        guidance_frequency = max(1, int(guidance_frequency))

        if loss_fn not in ("slice_pred", None, "none"):
            raise ValueError("Only 'slice_pred' is supported")

        if (additional_inputs is not None) and (loss_fn == "slice_pred"):
            if not hasattr(self, "vjepa_encoder") or not hasattr(self, "vjepa_predictor") or not hasattr(self, "vjepa_target_encoder"):
                vjepa_variant: str = additional_inputs.get("vjepa_variant", "vit_giant")
                img_size: int = int(additional_inputs.get("vjepa_img_size", 256 if vjepa_variant != "vit_giant_384" else 384))
                hub_fn_map = {
                    "vit_large": "vjepa2_vit_large",
                    "vit_huge": "vjepa2_vit_huge",
                    "vit_giant": "vjepa2_vit_giant",
                    "vit_giant_384": "vjepa2_vit_giant_384",
                }
                hub_fn_name = hub_fn_map.get(vjepa_variant, "vjepa2_vit_giant")
                enc, pred = torch.hub.load("facebookresearch/vjepa2", hub_fn_name)
                vjepa_dtype_str = str(additional_inputs.get("vjepa_dtype", "fp32")).lower()
                desired_dtype = torch.float32 if vjepa_dtype_str != "bf16" else torch.bfloat16
                self.vjepa_encoder = enc.to(self._execution_device, dtype=desired_dtype).eval()
                self.vjepa_target_encoder = copy.deepcopy(self.vjepa_encoder).eval()
                self.vjepa_predictor = pred.to(self._execution_device, dtype=desired_dtype).eval()
                # Freeze VJEPA module weights to avoid computing parameter grads while
                # still allowing gradients to flow w.r.t. inputs (the decoded frames)
                for m in (self.vjepa_encoder, self.vjepa_target_encoder, self.vjepa_predictor):
                    for p in m.parameters():
                        p.requires_grad_(False)
                self.vjepa_resize = T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
                self.vjepa_normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            masking_mode: str = str(additional_inputs.get("vjepa_masking_mode", "causal")).lower()
            context_frames: int = int(additional_inputs.get("vjepa_context_frames", 15))
            mask_ratio: float = float(additional_inputs.get("vjepa_mask_ratio", 0.75))
            slice_window_size: int = int(additional_inputs.get("slice_window_size", 16))
            slice_stride: int = int(additional_inputs.get("slice_stride", 2))

        # Optional: save intermediate decoded videos for visualization
        save_intermediate = additional_inputs.get("save_intermediate", False) if additional_inputs else False
        save_dir = None

        if save_intermediate:
            save_dir = additional_inputs.get("intermediate_save_dir", None) if additional_inputs else None
            if save_dir is None:
                save_dir = os.path.join("results", "intermediate_cosmos", datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(save_dir, exist_ok=True)

        def _save_intermediate_video(decoded_video_bcthw: torch.Tensor, path: str, fps: int = 8) -> None:
            x = decoded_video_bcthw.detach().cpu()
            x = ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
            _, C, T, H, W = x.shape
            frames = []
            for tt in range(T):
                frame = (x[0, :, tt] * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
                frames.append(Image.fromarray(frame))
            
            # Save as MP4 (more efficient than individual PNGs)
            export_to_video(frames, path, fps=fps)
            
            # Clear the tensor and frames from memory immediately
            del x, frames
            torch.cuda.empty_cache()

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]

                # Determine if we are in the guidance range
                in_guidance_range = (
                    (loss_fn == "slice_pred")
                    and (additional_inputs is not None)
                    and (guidance_step[i] != 0)
                    and ((i % max(1, int(guidance_frequency))) == 0)
                )
                n_repeats = guidance_step[i]
                ctx = torch.enable_grad() if in_guidance_range else torch.no_grad()

                for rep in range(n_repeats + 1):
                    # Set requires_grad only when within the guidance range
                    latents = latents.detach().requires_grad_(in_guidance_range)

                    # Conditional path
                    cond_latent = latents * c_in
                    cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                    cond_latent = cond_latent.to(transformer_dtype)
                    cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                    cond_timestep = cond_timestep.to(transformer_dtype)

                    # (2) Predict noise: two-pass (conditional + unconditional) to match Cosmos shapes

                    with ctx:
                        model_out_cond = self.transformer(
                            hidden_states=cond_latent,
                            timestep=cond_timestep,
                            encoder_hidden_states=prompt_embeds,
                            fps=fps,
                            condition_mask=cond_mask,
                            padding_mask=padding_mask,
                            return_dict=False,
                        )[0]

                        # Predicted clean from conditional path (keep full grad path to latents)
                        pure_cond = (c_skip * latents + c_out * model_out_cond.float()).to(transformer_dtype)
                        # Blended version only for denoising/scheduler
                        noise_pred_cond = cond_indicator * conditioning_latents + (1 - cond_indicator) * pure_cond

                    # with ctx:
                        if self.do_classifier_free_guidance:
                            uncond_latent = latents * c_in
                            uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                            uncond_latent = uncond_latent.to(transformer_dtype)
                            uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                            uncond_timestep = uncond_timestep.to(transformer_dtype)

                            
                            model_out_uncond = self.transformer(
                                hidden_states=uncond_latent,
                                timestep=uncond_timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                fps=fps,
                                condition_mask=uncond_mask,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]

                            noise_pred_uncond = (c_skip * latents + c_out * model_out_uncond.float()).to(transformer_dtype)
                            noise_pred_uncond = (
                                uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                            )

                            noise_pred = noise_pred_cond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_cond

                        # print_gpu_memory(True, "After transformer")

                    # (6) If in guidance range, compute MSE loss and gradient
                    if in_guidance_range and rep < n_repeats:

                        # Convert predicted clean latents to VAE space and optionally decode at a lower spatial scale
                        # Ensure gradients are enabled for decode + loss so that total_loss depends on latents
                        with torch.enable_grad():
                            # Use noise_pred for gradient computation (same as working cosmos2_multiframe2.py)
                            # pred_clean = noise_pred.requires_grad_(True)
                            pred_clean = noise_pred
                            # Cast pred_clean to VAE dtype for decoding path
                            pred_clean = pred_clean.to(self.vae.dtype)
                            latents_mean = (
                                torch.tensor(self.vae.config.latents_mean)
                                .view(1, self.vae.config.z_dim, 1, 1, 1)
                                .to(pred_clean.device, pred_clean.dtype)
                            )
                            latents_std = (
                                torch.tensor(self.vae.config.latents_std)
                                .view(1, self.vae.config.z_dim, 1, 1, 1)
                                .to(pred_clean.device, pred_clean.dtype)
                            )
                            pred_for_decode = pred_clean * latents_std / self.scheduler.config.sigma_data + latents_mean

                            vae_decode_scale = 1.0
                            if additional_inputs is not None:
                                vae_decode_scale = float(additional_inputs.get("vae_decode_scale", 1.0))

                            if vae_decode_scale != 1.0:
                                # Down/up-sample only spatial dims (keep temporal length)
                                pred_for_decode = F.interpolate(
                                    pred_for_decode,
                                    # scale_factor=(1.0, vae_decode_scale, vae_decode_scale),
                                    scale_factor=(1, 0.5, 0.5),
                                    mode="trilinear",
                                    align_corners=False,
                                )
                            # print("pred_for_decode", pred_for_decode.shape)


                            # print("pred_for_decode", pred_for_decode.shape)

                            # For a tensor of shape [1, 16, 24, 88, 160], chunk it into size 4 along the third dim (dim=2)
                            # Example: chunks = torch.chunk(tensor, chunks=24//4, dim=2)
                            latents_chunks = torch.chunk(pred_for_decode, chunks=pred_for_decode.shape[2] // 4, dim=2)
                            # print("latents_chunks", len(latents_chunks))
                            print("latents_chunks", len(latents_chunks), latents_chunks[0].shape)
                            
                            # with torch.no_grad():
                            # for chunk in latents_chunks:
                            def decode_evaluate_chunk(pred_for_decode):
                                vae_input = pred_for_decode.to(self.vae.dtype)
                                decoded_full = self.vae.decode(vae_input, return_dict=False)[0]
                                # print("decoded_full", decoded_full.shape)

                                # Sliding-window predictor loss over temporal slices on decoded frames
                                B, C, Tdec, Hdec, Wdec = decoded_full.shape
                                x_bcthw = ((decoded_full.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
                                
                                # print("x_bcthw", x_bcthw.shape)
                                # Get V-JEPA encoder dtype for consistent processing
                                device_sw = decoded_full.device
                                enc_dtype = next(self.vjepa_encoder.parameters()).dtype
                                
                                frames_proc: List[torch.Tensor] = []
                                for tt in range(Tdec):
                                    x_chw = x_bcthw[0, :, tt]
                                    x_chw = self.vjepa_resize(x_chw)
                                    x_chw = self.vjepa_normalize(x_chw)
                                    frames_proc.append(x_chw)
                                clip_cthw = torch.stack(frames_proc, dim=1).unsqueeze(0)  # [1,C,T,H,W]
                                # print("clip_cthw", clip_cthw.shape)
                                
                                # Convert to V-JEPA encoder dtype to avoid mismatch
                                clip_cthw = clip_cthw.to(device_sw, dtype=enc_dtype)
                                print("clip_cthw", clip_cthw.shape)

                                # Determine windowing
                                WSIZE = max(1, min(slice_window_size, clip_cthw.size(2)))
                                STRIDE = max(1, slice_stride)

                                # Tokenization geometry
                                img_size_local = self.vjepa_resize.size[0] if isinstance(self.vjepa_resize.size, tuple) else self.vjepa_resize.size
                                grid_size = img_size_local // getattr(self.vjepa_encoder, "patch_size", 16)
                                tube = int(getattr(self.vjepa_encoder, "tubelet_size", 2))

                                starts = list(range(0, clip_cthw.size(2) - WSIZE + 1, STRIDE))
                                if len(starts) == 0:
                                    starts = [0]
                                # Skip windows fully inside conditioned temporal region to keep grad path
                                cond_latent_frames = int(cond_indicator[0, 0].sum().item())
                                cond_cutoff_dec = max(0, (cond_latent_frames - 1) * self.vae_scale_factor_temporal + 1)
                                starts = [s for s in starts if s + WSIZE > cond_cutoff_dec]
                                if len(starts) == 0:
                                    starts = [cond_cutoff_dec]
                                

                                chunk = clip_cthw
                                Tchunk = chunk.size(2)
                                grid_depth = int(Tchunk // tube)

                                ctxt_positions, tgt_positions = generate_vjepa_masks(
                                    masking_mode=masking_mode,
                                    batch_size=chunk.shape[0],
                                    img_size=img_size_local,
                                    frames_per_clip=Tchunk,
                                    encoder=self.vjepa_encoder,
                                    context_frames=7,
                                    mask_ratio=mask_ratio,
                                    device=device_sw,
                                )

                                def forward_target(c):
                                    with torch.no_grad():
                                        h = self.vjepa_target_encoder(c)
                                        h = torch.stack([F.layer_norm(hi, (hi.size(-1),)) for hi in h])
                                    return h.detach()

                                def forward_context(c):
                                    # Enable gradients to flow to input chunk
                                    z = self.vjepa_encoder(c, ctxt_positions)
                                    z = self.vjepa_predictor(z, ctxt_positions, tgt_positions)
                                    z = F.layer_norm(z, (z.size(-1),))
                                    return z

                                def loss_fn_v2(z, h):
                                    h = apply_masks(h, tgt_positions, concat=False)
                                    loss = 1 - F.cosine_similarity(z, h[0], dim=1).mean()
                                    return loss

                                h_tokens = forward_target(chunk)
                                z_pred = forward_context(chunk)

                                z_chunk = z_pred.to(h_tokens.device)
                                chunk_loss = loss_fn_v2(z_chunk, h_tokens)



                                return chunk_loss
                            
                            with torch.no_grad():
                                chunk_losses = [decode_evaluate_chunk(chunk) for chunk in latents_chunks]

                            print("chunk_losses", chunk_losses)
                            max_idx = torch.stack(chunk_losses).argmax().item()
                            

                            with torch.enable_grad():
                                max_chunk = latents_chunks[max_idx]
                                total_loss = decode_evaluate_chunk(max_chunk)
                            
                            # grad = torch.autograd.grad(total_loss, latents, retain_graph=False, create_graph=False)[0]
                        # print("total_loss", total_loss.requires_grad)
                        # print("latents", latents.requires_grad)
                        # print("pred_clean", pred_clean.requires_grad)
                        # print_gpu_memory(True, "Before total_loss.backward")

                        total_loss.backward()
                        grad = latents.grad.clone()
                        latents.grad = None  # Clear the gradients

                        # grad = torch.autograd.grad(total_loss, latents, retain_graph=False, create_graph=False)[0]
                        # print_gpu_memory(True, "After total_loss.backward")


                        if (i + shorten_steps) > travel_time[0] and (i + shorten_steps) < travel_time[1]:
                            print("Time travel!")
                            with torch.no_grad():
                                
                                # noise = randn_tensor(pred_clean.shape, generator=generator, device=device, dtype=pred_clean.dtype)
                                # pred_clean = pred_clean - guidance_lr[i] * rho * grad
                                # latents = current_sigma * noise + (1 - current_sigma) * pred_clean
                                
                                noise = randn_tensor(pred_clean.shape, generator=generator, device=device, dtype=pred_clean.dtype)
                                latents = current_sigma * noise + (1 - current_sigma) * pred_clean
                                latents = latents - guidance_lr[i]  * grad # update with guidance
                            
                        else:
                            def log_grad_spread(g, delta_base, step_i, sample_idx: int = 0, topk: int = 5):
                                """
                                g            : [B,C,T,H,W]   (∂L/∂x_t)
                                delta_base   : [B,C,T,H,W]   (vanilla solver step per frame)
                                step_i       : int           (k in your loop)
                                sample_idx   : which batch element to print
                                topk         : how many top frames to summarize
                                """
                                print('g', g.shape)
                                print('delta_base', delta_base.shape)
                                eps = 1e-12
                                assert g.dim() == 5 and delta_base.shape == g.shape, "shape mismatch"
                                B, C, T, H, W = g.shape
                                b = min(sample_idx, B-1)

                                # Per-frame L2 norms over C,H,W
                                reduce_chw = (1, 3, 4)
                                g_t = g.pow(2).sum(dim=reduce_chw).sqrt()                   # [B,T]
                                d_t = delta_base.pow(2).sum(dim=reduce_chw).sqrt()          # [B,T]
                                dot_t = (g * delta_base).sum(dim=reduce_chw)                # [B,T]
                                cos_t = dot_t / (g_t * d_t + eps)                           # [B,T] per-frame cosine

                                # Normalized energy distribution over frames
                                p_t = g_t / (g_t.sum(dim=1, keepdim=True) + eps)            # [B,T], sum_t p_t = 1

                                # Concentration metrics
                                hhi = (p_t**2).sum(dim=1)                                   # Herfindahl index
                                eff_frames = 1.0 / (hhi + eps)                              # "effective number of frames"

                                # How many frames cover 50% / 90% of the mass?
                                p_sorted, idx_sorted = torch.sort(p_t[b], descending=True)
                                csum = torch.cumsum(p_sorted, dim=0)
                                k50 = int((csum >= 0.50).nonzero(as_tuple=False)[0]) + 1
                                k90 = int((csum >= 0.90).nonzero(as_tuple=False)[0]) + 1

                                # Top-k summary
                                K = min(topk, T)
                                top_idx = idx_sorted[:K]
                                top_mass = p_sorted[:K].sum().item()
                                top_cos_mean = cos_t[b, top_idx].mean().item()

                                # Compact, readable printout
                                tops = [(int(i), float(p_t[b, i]), float(cos_t[b, i])) for i in top_idx]
                                print(f"[k={step_i}] grad spread (b={b}): eff_frames={eff_frames[b].item():.2f}, "
                                    f"k50={k50}, k90={k90}, top{K}_mass={top_mass:.2f}, top{K}_cos={top_cos_mean:.3f}")
                                print(f"          top{K} frames (idx, p_t, cos): {tops}")

                                # Optional: an ASCII bar for p_t (one character per frame, scaled)
                                width = 30
                                bars = ''.join('█' * max(1, int(width * float(p))) for p in p_t[b].tolist())
                                print(f"          p_t bars (T={T}): {bars}")

                            print("DLO!")
                            with torch.no_grad():
                                log_grad_spread(grad, (noise_pred_cond - noise_pred_uncond), step_i=i, sample_idx=0, topk=5)
                                eps = 1e-8
                                scale = guidance_lr[i] * (latents.norm(2) / (grad.norm(2) + eps))
                                # scale = guidance_lr[i] * (1 / grad.norm(2))
                                print("scale", scale)
                                latents = latents - scale * grad


                        torch.cuda.empty_cache()


                # Convert to eps and take scheduler step using last_pred_clean
                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            if self.safety_checker is not None:
                self.safety_checker.to(device)
                video = self.video_processor.postprocess_video(video, output_type="np")
                video = (video * 255).astype(np.uint8)
                video_batch = []
                for vid in video:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
                video = self.video_processor.postprocess_video(video, output_type=output_type)
                self.safety_checker.to("cpu")
            else:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)
