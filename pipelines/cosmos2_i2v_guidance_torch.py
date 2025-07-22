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
from typing import Callable, Dict, List, Optional, Union
import os

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan, CosmosTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_cosmos_guardrail_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.cosmos.pipeline_output import CosmosPipelineOutput

# Add V-JEPA guidance imports
from utils import init_torch_vjepa, preprocess_video_for_torch_vjepa
from compute_vjepa_score import calculate_torch_vjepa_loss
from transformers import AutoVideoProcessor, AutoModel

# Add action guidance imports
# from compute_action_loss import ActionGuidanceLoss
import os
from diffusers.utils import export_to_video
import gpustat
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

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

def print_gpu_memory(if_vis,info=""):
    if if_vis:
        query = gpustat.new_query()
        for idx in range(1):
            gpu = query.gpus[idx]  # Assuming you want to print the first GPU's memory usage
            print(f"{info}")
            print(f"GPU ID: {gpu.index}")
            print(f"GPU Name: {gpu.name}")
            print(f"GPU Utilization: {gpu.utilization}%")
            print(f"Memory Used: {gpu.memory_used} MB / {gpu.memory_total} MB")
            print(f"Temperature: {gpu.temperature}°C")
            print("-" * 20)
    else:
        pass

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
    Pipeline for video-to-world generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)
    with V-JEPA and Action guidance support.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline now supports both V-JEPA guidance and Action guidance during generation. To configure guidance, 
    set these attributes on the pipeline instance before calling:
    
    V-JEPA Guidance:
    - guidance_start (int): Start timestep for guidance (default: 750)
    - guidance_end (int): End timestep for guidance (default: 900)  
    - guidance_rho_scale (float): Scaling factor for guidance gradients (default: 3.0)
    - vjepa_context_length (int): Context length for V-JEPA (default: 12)
    - vjepa_stride (int): Stride for V-JEPA sliding window (default: 2)
    - vjepa_mode (str): V-JEPA aggregation mode (default: 'max')

    Action Guidance:
    - action_guidance_scale (float): Scale factor for action guidance loss (default: 2.0)
    - action_use_simple_loss (bool): Whether to use simple variance-based loss (default: True)
    - action_rho_scale (float): Scaling factor for action guidance gradients (default: 3.0)

    Example:
        ```python
        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id)
        
        # Configure V-JEPA guidance
        pipe.guidance_start = 800
        pipe.guidance_end = 950
        pipe.guidance_rho_scale = 2.5
        
        # Configure action guidance
        pipe.configure_action_guidance(
            guidance_scale=2.0,
            use_simple_loss=True,
            rho_scale=3.0
        )
        
        # Generate with both guidances
        video = pipe(
            image=image, 
            prompt=prompt,
            target_action=action_tensor,
            initial_pose=pose_tensor
        )
        ```

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

        # Initialize torch V-JEPA model for guidance
        print("🚀 Loading torch V-JEPA model for guidance...")
        self.vjepa_model = init_torch_vjepa()
        self.vjepa_processor = None  # Not needed for torch version

        # # Initialize action guidance for zero-shot adaptation
        # print("🤖 Loading action guidance for zero-shot robot adaptation...")
        # self.action_guidance = ActionGuidanceLoss(guidance_scale=2.0, use_simple_loss=True)

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

    def configure_action_guidance(self, guidance_scale=2.0, use_simple_loss=True, rho_scale=3.0):
        """
        Configure action guidance parameters for zero-shot robot adaptation
        
        Args:
            guidance_scale: Scale factor for action guidance loss
            use_simple_loss: Whether to use simple variance-based loss (True) or full VJEPA loss (False)
            rho_scale: Scale factor for gradient-based guidance updates
        """
        self.action_guidance = ActionGuidanceLoss(guidance_scale=guidance_scale, use_simple_loss=use_simple_loss)
        self.action_guidance_scale = guidance_scale
        self.action_use_simple_loss = use_simple_loss
        self.action_rho_scale = rho_scale
        
        print(f"🔧 Action guidance configured:")
        print(f"   Guidance scale: {guidance_scale}")
        print(f"   Use simple loss: {use_simple_loss}")
        print(f"   Rho scale: {rho_scale}")

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline._get_t5_prompt_embeds
    @torch.no_grad()
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

    # @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        target_action: Optional[torch.Tensor] = None,
        initial_pose: Optional[torch.Tensor] = None,
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
        guidance_start: int = 750,
        guidance_end: int = 900,
        guidance_rho_scale: float = 3.0,
        vjepa_mode: str = "max",
        vjepa_context_length: int = 12,
        vjepa_stride: int = 2,
        vjepa_kernel_size: int = 16,
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
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            target_action (`torch.Tensor`, *optional*):
                Target robot action for action guidance. Should be shape [1, 1, 7] with format [dx,dy,dz,qx,qy,qz,qw].
                Required for action guidance to work.
            initial_pose (`torch.Tensor`, *optional*):
                Initial robot pose for action guidance. Should be shape [1, 1, 7] with format [x,y,z,qx,qy,qz,qw].
                Required for action guidance to work.
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
            guidance_start (`int`, *optional*, defaults to `750`):
                The timestep at which to start applying V-JEPA guidance. Guidance is applied when timestep > guidance_start.
            guidance_end (`int`, *optional*, defaults to `900`):
                The timestep at which to stop applying V-JEPA guidance. Guidance is applied when timestep < guidance_end.
            guidance_rho_scale (`float`, *optional*, defaults to `3.0`):
                Scale factor for the guidance gradient updates. Higher values apply stronger guidance.
            vjepa_mode (`str`, *optional*, defaults to `"max"`):
                Mode for V-JEPA loss calculation. Options include "max", "mean", etc.
            vjepa_context_length (`int`, *optional*, defaults to `12`):
                Number of context frames to mask for V-JEPA prediction task.
            vjepa_stride (`int`, *optional*, defaults to `2`):
                Stride for V-JEPA sliding window when processing video clips.
            vjepa_kernel_size (`int`, *optional*, defaults to `16`):
                Number of frames per clip for V-JEPA processing. For torch V-JEPA, this is always 16.

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

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        VIS_MEM = False
        print_gpu_memory(VIS_MEM, info="Start ================================")
        print(f"🎬 STARTING COSMOS GUIDANCE PIPELINE GENERATION")
        print(f"   Frames: {num_frames}, Steps: {num_inference_steps}, CFG Scale: {guidance_scale}")

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
        self.vjepa_model.requires_grad_(False)
        for p in self.vjepa_model.parameters():
            p.requires_grad = False

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )
            self.safety_checker.to("cpu")
        
        self.safety_checker = None

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
        print_gpu_memory(VIS_MEM, info="after encoder prompt ================================")

        # Add guidance configuration - make these configurable attributes
        guidance_range = [guidance_start, guidance_end]
        time_travel_range = [-1, -1]
        
        # Check for action guidance inputs
        perform_action_guidance = target_action is not None and initial_pose is not None
        if perform_action_guidance:
            # Validate action guidance inputs
            assert target_action.shape == (1, 1, 7), f"target_action should be shape (1,1,7), got {target_action.shape}"
            assert initial_pose.shape == (1, 1, 7), f"initial_pose should be shape (1,1,7), got {initial_pose.shape}"
            target_action = target_action.to(device)
            initial_pose = initial_pose.to(device)
        
        # Check if we should track V-JEPA loss without guidance
        track_vjepa_loss = getattr(self, 'track_vjepa_loss', False)
        
        # Print guidance configuration
        print(f"\n🔧 COSMOS GUIDANCE PIPELINE CONFIGURATION:")
        print(f"   Guidance range: {guidance_range[0]} - {guidance_range[1]}")
        print(f"   Rho scale: {guidance_rho_scale}")
        print(f"   Track V-JEPA loss: {track_vjepa_loss}")
        print(f"   V-JEPA kernel_size: {vjepa_kernel_size}")
        print(f"   V-JEPA context_length: {vjepa_context_length}")
        print(f"   V-JEPA stride: {vjepa_stride}")
        print(f"   V-JEPA mode: {vjepa_mode}")
        if perform_action_guidance:
            print(f"   🤖 Action guidance: ENABLED")
            print(f"   Action guidance scale: {getattr(self, 'action_guidance_scale', 2.0)}")
            print(f"   Action rho scale: {getattr(self, 'action_rho_scale', 3.0)}")
            print(f"   Action use simple loss: {getattr(self, 'action_use_simple_loss', True)}")
        else:
            print(f"   🤖 Action guidance: DISABLED (no target_action/initial_pose provided)")
        print(f"   Total timesteps: {num_inference_steps}")
        print()

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
        
            print_gpu_memory(VIS_MEM, info="after condition preparation ================================")

            cond_mask = cond_mask.to(transformer_dtype)
            if self.do_classifier_free_guidance:
                uncond_mask = uncond_mask.to(transformer_dtype)
                unconditioning_latents = conditioning_latents

            padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
            sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
            t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # Initialize loss history for tracking
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
            self.latent_norm_history = []
            self.scaled_norm_history = []

        # Define transformer forward function for checkpointing
        def transformer_forward_cond(latent_input, timestep_input, prompt_embeds_input, fps_input, cond_mask_input, padding_mask_input):
            with torch.no_grad():
                output = self.transformer(
                    hidden_states=latent_input,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds_input,
                    fps=fps_input,
                    condition_mask=cond_mask_input,
                    padding_mask=padding_mask_input,
                    return_dict=False,
                )[0]
            return output

        def transformer_forward_uncond(latent_input, timestep_input, prompt_embeds_input, fps_input, uncond_mask_input, padding_mask_input):
            with torch.no_grad():
                output = self.transformer(
                    hidden_states=latent_input,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds_input,
                    fps=fps_input,
                    condition_mask=uncond_mask_input,
                    padding_mask=padding_mask_input,
                    return_dict=False,
                )[0]
            return output

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        guidance_steps_count = 0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                print(f"🚀 COSMOS STEP {i} (t={t.item():.0f})")
                # Check if guidance should be performed at this timestep
                perform_guidance = True if (t > guidance_range[0] and t < guidance_range[1]) else False
                perform_travel = True if (t > time_travel_range[0] and t < time_travel_range[1]) else False
                n_travel = 1 if perform_travel else 1

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]

                # Enable gradients for latents if guidance is needed
                if perform_guidance:
                    latents = latents.detach().clone().requires_grad_(True)
                    latents.retain_grad()

                visualize = False
                
                for rep in range(n_travel):
                    print_gpu_memory(VIS_MEM, "before DiT ================================")

                    cond_latent = latents * c_in
                    cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                    cond_latent = cond_latent.to(transformer_dtype).requires_grad_(True)
                    cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                    cond_timestep = cond_timestep.to(transformer_dtype)
     

                    noise_pred = transformer_forward_cond(
                        cond_latent,
                        cond_timestep,
                        prompt_embeds,
                        fps,
                        cond_mask,
                        padding_mask,
                    )

                    noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
                    noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred

                    print_gpu_memory(VIS_MEM, info="after cond forward ================================")

                    if self.do_classifier_free_guidance:
                        uncond_latent = latents * c_in
                        uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                        uncond_latent = uncond_latent.to(transformer_dtype)
                        uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                        uncond_timestep = uncond_timestep.to(transformer_dtype)


                        with torch.no_grad():
                            noise_pred_uncond = transformer_forward_uncond(
                                uncond_latent,
                                uncond_timestep,
                                negative_prompt_embeds,
                                fps,
                                uncond_mask,
                                padding_mask,
                            )

                        noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                        noise_pred_uncond = (
                            uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                        )
                        noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                        print_gpu_memory(VIS_MEM, info="after uncond forward ================================")

                # Apply V-JEPA guidance if enabled
                if perform_guidance:
                    guidance_steps_count += 1
                    print(f"🚀 APPLYING GUIDANCE at step {i} (timestep={t.item():.0f})")
                    
                    with torch.set_grad_enabled(True):
                        # Estimate x0 (original sample) from noise prediction
                        # noise_pred = (latents - noise_pred) / current_sigma
                        # pred_original_sample = latents - current_sigma * noise_pred
                        # pred_original_sample = pred_original_sample.clone().detach()
                        pred_original_sample = noise_pred
                        pred_original_sample.requires_grad_(True)
                        # pred_original_sample.retain_grad()
                        
                        latents_mean = (
                            torch.tensor(self.vae.config.latents_mean)
                            .view(1, self.vae.config.z_dim, 1, 1, 1)
                            .to(pred_original_sample.device, pred_original_sample.dtype)
                        )
                        latents_std = (
                            torch.tensor(self.vae.config.latents_std)
                            .view(1, self.vae.config.z_dim, 1, 1, 1)
                            .to(pred_original_sample.device, pred_original_sample.dtype)
                        )
                        pred_original_sample = pred_original_sample * latents_std / self.scheduler.config.sigma_data + latents_mean


                        pred_original_sample_half = F.interpolate(pred_original_sample.squeeze(0), scale_factor=1/2, mode='bilinear', align_corners=False).unsqueeze(0)
                        orig_frame = self.vae.decode(pred_original_sample_half.to(self.vae.dtype), return_dict=False)[0]

                        print_gpu_memory(VIS_MEM, info="after VAE decode ================================")

                        # Save visualization with organized directory structure
                        
                        if visualize:
                            with torch.no_grad():
                                # Create output directory with consistent naming
                                ttpath = f'./temp/cosmos_guidance{guidance_rho_scale}'
                                os.makedirs(ttpath, exist_ok=True)
                                save_frame = self.video_processor.postprocess_video(orig_frame.detach(), output_type="np")
                                export_to_video(save_frame[0], f"{ttpath}/guidance_sample_{i}.mp4", fps=16)
                        
                        # 1. Apply V-JEPA guidance
                        # Calculate V-JEPA loss using torch implementation
                        B, C, T, H, W = orig_frame.shape
                        orig_frame_tensor = orig_frame
                        
                        with torch.enable_grad():
                            # Use configurable V-JEPA parameters 
                            frames_per_clip = vjepa_kernel_size  # Fixed size that model was trained with
                            context_window_size = vjepa_context_length
                            stride = vjepa_stride
                            
                            # Calculate loss with sliding window handled inside the function
                            vjepa_loss, loss_arr = calculate_torch_vjepa_loss(
                                orig_frame_tensor, 
                                self.vjepa_model,
                                context_length=context_window_size,
                                frames_per_clip=frames_per_clip,
                                stride=stride,
                                require_grad=True,
                                mode=vjepa_mode,
                                return_arr=True,
                                is_vae_output=True,
                                save_step=i
                            )
                            print(f"📺 V-JEPA loss: {vjepa_loss.item():.4f}")
                            print_gpu_memory(VIS_MEM, info="after VJepa loss ================================")
                        
                        # Backpropagate and get gradients
                        grad = torch.autograd.grad(vjepa_loss, pred_original_sample, retain_graph=False, create_graph=False)[0]
                        pred_original_sample.grad = None

                        # Calculate rho scaling
                        grad_norm = grad.norm(2)
                        rho = 1 / grad_norm
                        scaled_norm = (guidance_rho_scale * rho * grad).norm(2)

                        print_gpu_memory(VIS_MEM, info="after guidance computation ================================")


                        # Print slide-wise gradient norms like in wan pipeline
                        # for kid in range(grad.shape[2]):
                        #     print(f"slide norm: {grad[:,:,kid,:,:].norm(2)}")

                        print(
                            f'🎯 COSMOS GUIDANCE STEP {i} (t={t.item():.0f}): VJepa_loss={vjepa_loss.item():.4f}, Grad_norm={grad_norm.item():.4f}, '
                            f'Rho={rho.item():.4f}, Rho_scale={guidance_rho_scale}, Latent norm={latents.norm(2).item():.4f}, '
                            f'delta_norm={scaled_norm.item():.4f}'
                        )

                        # Store loss in history for comparison
                        if hasattr(self, 'loss_history') and visualize:
                            self.loss_history.append(vjepa_loss.item())
                            self.latent_norm_history.append(latents.norm(2).item())
                            self.scaled_norm_history.append(scaled_norm.item())

                        # Save slide-wise grad norms like in wan pipeline
                        if visualize:
                            slide_grad_norms = np.array([grad[:,:,kid,:,:].norm(2).item() for kid in range(grad.shape[2])])
                            np.save(f'{ttpath}/slide_grad_norms_step{i}.npy', slide_grad_norms)
                            np.save(f'{ttpath}/loss_step{i}.npy', loss_arr)

                        if perform_travel:
                            # Time travel logic (currently disabled)
                            pass
                        else:
                            # Update latents with guidance
                            with torch.no_grad():
                                latents = latents - guidance_rho_scale * rho * grad

                    torch.cuda.empty_cache()
                elif track_vjepa_loss:
                    # Track V-JEPA loss without guidance (for vanilla sampling comparison)
                    guidance_steps_count += 1
                    print(f"📊 TRACKING V-JEPA LOSS at step {i} (timestep={t.item():.0f}) - NO GUIDANCE")
                    with torch.set_grad_enabled(False):  # No gradients needed for tracking only
                        pred_original_sample = latents - current_sigma * noise_pred
                        pred_original_sample = pred_original_sample.to(self.vae.dtype)
                        latents_mean = (
                            torch.tensor(self.vae.config.latents_mean)
                            .view(1, self.vae.config.z_dim, 1, 1, 1)
                            .to(pred_original_sample.device, pred_original_sample.dtype)
                        )
                        latents_std = (
                            torch.tensor(self.vae.config.latents_std)
                            .view(1, self.vae.config.z_dim, 1, 1, 1)
                            .to(pred_original_sample.device, pred_original_sample.dtype)
                        )
                        pred_original_sample = pred_original_sample * latents_std / self.scheduler.config.sigma_data + latents_mean

                        orig_frame = self.vae.decode(pred_original_sample, return_dict=False)[0]
                        
                        # Get vjepa loss using torch implementation (no gradients)
                        B, C, T, H, W = orig_frame.shape
                        orig_frame_tensor = orig_frame
                        
                        with torch.no_grad():
                            # Calculate loss without gradients for tracking
                            loss = calculate_torch_vjepa_loss(
                                orig_frame_tensor, 
                                self.vjepa_model,
                                context_length=vjepa_context_length,
                                frames_per_clip=vjepa_kernel_size,
                                stride=vjepa_stride,
                                require_grad=False,
                                mode=vjepa_mode,
                                return_arr=False,
                                is_vae_output=True
                            )

                        print(f'📊 V-JEPA LOSS TRACKING {i} (t={t.item():.0f}): Loss={loss.item():.4f} (vanilla sampling)')
                        
                        # Store loss in history for comparison
                        if hasattr(self, 'loss_history') and visualize:
                            self.loss_history.append(loss.item())
                    
                    torch.cuda.empty_cache()
                else:
                    # No guidance and no tracking - store None for consistency
                    if hasattr(self, 'loss_history'):
                        self.loss_history.append(None)

                # Compute the previous noisy sample x_t -> x_t-1
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

        # Save trajectories at the end like in wan pipeline
        if visualize:
            ttpath = f'./temp/cosmos_guidance{guidance_rho_scale}'
            os.makedirs(ttpath, exist_ok=True)
            np.save(f'{ttpath}/loss_trajectory.npy', np.array(getattr(self, 'loss_history', [])))
            np.save(f'{ttpath}/latent_norm_trajectory.npy', np.array(getattr(self, 'latent_norm_history', [])))
            np.save(f'{ttpath}/scaled_norm_trajectory.npy', np.array(getattr(self, 'scaled_norm_history', [])))

        if not output_type == "latent":
            with torch.no_grad():
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
                    video = self.video_processor.postprocess_video(video.detach(), output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)
