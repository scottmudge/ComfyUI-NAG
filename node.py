import torch
import comfy
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
import latent_preview
import copy
from types import MethodType
from torch._dynamo.eval_frame import OptimizedModule

from .samplers import NAGCFGGuider as samplers_NAGCFGGuider
from .sample import sample_with_nag, sample_custom_with_nag
from .flux.model import NAGFluxSwitch
from .chroma.model import NAGChromaSwitch
from .sd.openaimodel import NAGUNetModelSwitch
from .sd3.mmdit import NAGOpenAISignatureMMDITWrapperSwitch
from .wan.model import NAGWanModelSwitch
from .hunyuan_video.model import NAGHunyuanVideoSwitch
from .hidream.model import NAGHiDreamImageTransformer2DModelSwitch
from .lumina2.model import NAGNextDiTSwitch

from comfy.ldm.flux.model import Flux
from comfy.ldm.chroma.model import Chroma
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
from comfy.ldm.wan.model import WanModel, VaceWanModel
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.hidream.model import HiDreamImageTransformer2DModel
from comfy.ldm.lumina.model import NextDiT


def common_ksampler_with_nag(model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name,
                             scheduler, positive, negative, nag_negative, latent, denoise=1.0, disable_noise=False,
                             start_step=None, last_step=None, force_full_denoise=False, **kwargs):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = sample_with_nag(
        model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, positive,
        negative, nag_negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
        seed=seed, **kwargs,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class NAGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            conditioning,
            nag_negative,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_sigma_end,
            latent_image,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(conditioning)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
        return (guider,)


class NAGCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive,
            negative,
            nag_negative,
            cfg,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_sigma_end,
            latent_image,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
        return (guider,)


class KSamplerWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "nag_negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler,
               positive, negative, nag_negative, latent_image, denoise=1.0):
        return common_ksampler_with_nag(model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
                                        sampler_name, scheduler, positive, negative, nag_negative, latent_image,
                                        denoise=denoise)


class KSamplerAdvancedWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(
            self, model, add_noise, noise_seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            sampler_name, scheduler, positive, negative, nag_negative,
            latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0,
    ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler_with_nag(
            model, noise_seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            sampler_name, scheduler, positive, negative, nag_negative,
            latent_image,
            denoise=denoise, disable_noise=disable_noise, start_step=start_at_step,
            last_step=end_at_step, force_full_denoise=force_full_denoise,
        )


class SamplerCustomWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "add_noise": ("BOOLEAN", {"default": True}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "nag_negative": ("CONDITIONING", {
                "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(
            self,
            model, add_noise, noise_seed, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            positive, negative, nag_negative,
            sampler, sigmas, latent_image,
    ):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = sample_custom_with_nag(
            model, noise, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            sampler, sigmas, positive, negative, nag_negative,
            latent_image,
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed,
        )

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)


class ModelNAG:
    """
    Applies NAG to a model using the NAGSwitch approach (similar to guider/ksampler technique).
    This patches the model's forward method to apply NAG during inference.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "nag_negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.001, 
                    "tooltip": "Strength of negative guidance effect"}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001, 
                    "tooltip": "Mixing coefficient that controls the balance between the normalized guided representation and the original positive representation."}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.001, 
                    "tooltip": "Clipping threshold that controls how much the guided attention can deviate from the positive attention."}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, 
                    "tooltip": "Sigma value at which NAG stops being applied. Set to 0.0 to apply NAG throughout the entire denoising process."}),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Applies Normalized Attention Guidance (NAG) to a model using the guider/ksampler technique. Works best for batch sizes up to 4. For larger or variable batch sizes, use NAGGuider or KSamplerWithNAG nodes. https://github.com/ChenDarYen/Normalized-Attention-Guidance"
    EXPERIMENTAL = True

    def patch(self, model, nag_negative, nag_scale, nag_alpha, nag_tau, nag_sigma_end):
        if nag_scale == 0:
            return (model,)
        
        # Clone the model to avoid modifying the original
        model_clone = model.clone()
        
        # Get the diffusion model
        diffusion_model = model_clone.model.diffusion_model
        if isinstance(diffusion_model, OptimizedModule):
            diffusion_model = diffusion_model._orig_mod
        
        # Determine model type and select appropriate NAGSwitch class
        model_type = type(diffusion_model)
        if model_type == Flux:
            switcher_cls = NAGFluxSwitch
        elif model_type == Chroma:
            switcher_cls = NAGChromaSwitch
        elif model_type == UNetModel:
            switcher_cls = NAGUNetModelSwitch
        elif model_type == OpenAISignatureMMDITWrapper:
            switcher_cls = NAGOpenAISignatureMMDITWrapperSwitch
        elif model_type in [WanModel, VaceWanModel]:
            switcher_cls = NAGWanModelSwitch
        elif model_type == HunyuanVideo:
            switcher_cls = NAGHunyuanVideoSwitch
        elif model_type == NextDiT:
            switcher_cls = NAGNextDiTSwitch
        elif model_type == HiDreamImageTransformer2DModel:
            switcher_cls = NAGHiDreamImageTransformer2DModelSwitch
        else:
            raise ValueError(
                f"Model type {model_type} is not supported for ModelNAG. "
                f"Supported types: Flux, Chroma, UNetModel, OpenAISignatureMMDITWrapper, "
                f"WanModel, VaceWanModel, HunyuanVideo, NextDiT, HiDreamImageTransformer2DModel"
            )
        
        # Store original nag_negative for dynamic batch size expansion
        # We don't expand it here - instead we'll expand it dynamically during forward pass
        original_nag_negative = copy.deepcopy(nag_negative)
        
        # Store original forward before any patching
        original_forward = diffusion_model.forward
        
        # Create a wrapper class that handles dynamic batch size expansion
        class DynamicNAGSwitch(switcher_cls):
            def __init__(self, model, original_nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end, original_forward, forward_wrapper):
                # Store original for dynamic expansion
                self._original_nag_negative = original_nag_negative
                self._nag_scale = nag_scale
                self._nag_tau = nag_tau
                self._nag_alpha = nag_alpha
                self._nag_sigma_end = nag_sigma_end
                self._model = model
                self._switcher = None
                self._current_batch_size = None
                self._original_forward = original_forward
                self._forward_wrapper = forward_wrapper
                
            def _expand_nag_negative(self, batch_size):
                """Expand nag_negative to match the given batch size"""
                if self._current_batch_size == batch_size and self._switcher is not None:
                    return  # Already expanded to correct size
                
                # Expand nag_negative to match batch size
                nag_negative_expanded = copy.deepcopy(self._original_nag_negative)
                nag_negative_expanded[0][0] = nag_negative_expanded[0][0].expand(batch_size, -1, -1)
                if nag_negative_expanded[0][1].get("pooled_output", None) is not None:
                    nag_negative_expanded[0][1]["pooled_output"] = nag_negative_expanded[0][1]["pooled_output"].expand(batch_size, -1)
                
                # If switcher exists, restore original forward first to clean up
                if self._switcher is not None:
                    # Temporarily restore original to clean up the old switcher's state
                    # But first, save the current wrapper if it exists
                    was_wrapped = hasattr(self._model, '_nag_wrapper_applied') and self._model._nag_wrapper_applied
                    if was_wrapped:
                        # Temporarily remove wrapper to get to the switcher's forward
                        self._model.forward = self._model._nag_actual_forward
                    self._switcher.set_origin()
                    # Now forward should be original_forward
                
                # Create new switcher with expanded batch size
                self._switcher = switcher_cls(
                    self._model,
                    nag_negative_expanded,
                    self._nag_scale, self._nag_tau, self._nag_alpha, self._nag_sigma_end,
                )
                self._switcher.set_nag()
                self._current_batch_size = batch_size
                
                # Store the switcher's forward BEFORE we wrap it
                # This is the actual NAG-patched forward we want to call
                switcher_forward = self._model.forward
                self._model._nag_actual_forward = switcher_forward
                
                # Re-apply our wrapper as the outermost wrapper
                if self._forward_wrapper is not None:
                    self._model.forward = MethodType(self._forward_wrapper, self._model)
                    self._model._nag_wrapper_applied = True
            
            def set_nag(self):
                # Don't set nag here - we'll do it dynamically in the forward wrapper
                pass
            
            def set_origin(self):
                if self._switcher is not None:
                    self._switcher.set_origin()
                    self._model.forward = self._original_forward
        
        # Create dynamic switcher (forward_wrapper will be set after switcher is created)
        dynamic_switcher = DynamicNAGSwitch(
            diffusion_model,
            original_nag_negative,
            nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            original_forward,
            None,  # Will be set below
        )
        
        # Define forward wrapper function that uses the dynamic switcher
        def nag_forward_wrapper(model_self, *args, **kwargs):
            # Get context to determine batch size
            context = None
            if 'context' in kwargs:
                context = kwargs['context']
            elif len(args) >= 3:
                # Context is typically the 3rd argument (after x, timestep)
                context = args[2]
            
            # Determine batch size from context or x
            batch_size = None
            if context is not None and isinstance(context, torch.Tensor):
                batch_size = context.shape[0]
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                # Fallback to x batch size
                batch_size = args[0].shape[0]
            
            # Expand nag_negative to match batch size if we have it
            if batch_size is not None and batch_size > 0:
                dynamic_switcher._expand_nag_negative(batch_size)
            
            # Call the actual forward method (stored by the switcher)
            return model_self._nag_actual_forward(*args, **kwargs)
        
        # Set the forward wrapper in the switcher
        dynamic_switcher._forward_wrapper = nag_forward_wrapper
        
        # Initialize with batch size 1 as default (will be expanded dynamically)
        # This will create the switcher and patch the forward, then wrap it
        dynamic_switcher._expand_nag_negative(1)
        
        # Store references
        model_clone._nag_switcher = dynamic_switcher
        model_clone._nag_original_forward = original_forward
        
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "NAGGuider": NAGGuider,
    "NAGCFGGuider": NAGCFGGuider,
    "KSamplerWithNAG": KSamplerWithNAG,
    "KSamplerWithNAG (Advanced)": KSamplerAdvancedWithNAG,
    "SamplerCustomWithNAG": SamplerCustomWithNAG,
    "ModelNAG": ModelNAG,
}
