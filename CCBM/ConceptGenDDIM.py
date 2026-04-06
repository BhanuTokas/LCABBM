import torch
from tqdm import tqdm
from PIL import Image
from typing import Union
import numpy as np
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from diffusers import DiffusionPipeline


def enable_memory_savings(pipe):
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    return pipe


def build_fixed_latent(
    pipe, height, width, device, latent_scale=8, seed=0, dtype=torch.float32
):
    """
    Build a fixed random latent tensor to pass to decoder.
    Assumes latent spatial dims = height/latent_scale, width/latent_scale.
    Returns: (latents, generator)
    """
    b = 1
    # infer channels from UNet if available
    unet = (
        getattr(pipe, "decoder", None)
        or getattr(pipe, "unet", None)
        or getattr(pipe, "unet", None)
    )
    if unet is None:
        raise RuntimeError(
            "Pipeline does not expose an accessible UNet (decoder/unet)."
        )
    in_ch = getattr(unet, "in_channels", None)
    if in_ch is None:
        # fallback channel for stable diffusion style latents
        in_ch = 4
    latent_h = height // latent_scale
    latent_w = width // latent_scale
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (b, in_ch, latent_h, latent_w),
        generator=torch_generator,
        device=device,
        dtype=dtype,
    )
    return latents, torch_generator


class CLIPInterface:

    def __init__(
        self,
        clip_model="openai/clip-vit-large-patch14",
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_un2clip: bool = False,
        un2clip_ckpt_path: str | None = None,
    ):
        """
        Parameters
        ----------
        clip_model       : HuggingFace model ID for the base CLIP model.
        dtype            : Model dtype.
        device           : Compute device.
        use_un2clip      : If True, patch the visual encoder with the un2CLIP
                           fine-tuned weights after loading the base CLIP model.
                           The text encoder is always kept as standard CLIP,
                           since un2CLIP only fine-tunes the visual encoder.
        un2clip_ckpt_path: Path to the un2CLIP .ckpt file (e.g.
                           './pretrained_models/openai_vit_l_14_224.ckpt').
                           Required when use_un2clip=True.
        """
        self.dtype = dtype
        self.device = device
        self.clip_model = clip_model
        self.use_un2clip = use_un2clip
        self.un2clip_ckpt_path = un2clip_ckpt_path

        if use_un2clip and un2clip_ckpt_path is None:
            raise ValueError(
                "un2clip_ckpt_path must be provided when use_un2clip=True."
            )

        self.initCLIP()

    def initCLIP(self):
        self.clip_model_default = self.clip_model
        self.clip_processor = CLIPImageProcessor.from_pretrained(self.clip_model)
        self.clip_image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(self.clip_model)
            .to(self.device)
            .eval()
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_model)
        self.text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(self.clip_model)
            .to(self.device)
            .eval()
        )

        # Patch the visual encoder with un2CLIP fine-tuned weights if requested.
        # un2CLIP only fine-tunes the vision transformer, so the text encoder
        # above is intentionally left as standard CLIP.
        if self.use_un2clip:
            self._load_un2clip_weights()

    def _load_un2clip_weights(self):
        """
        Load the un2CLIP checkpoint and patch the visual encoder's weights.

        The checkpoint stores only the visual encoder state_dict, matching the
        interface used in the official un2CLIP eval script (eval_mmvpvlm.py):
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.visual.load_state_dict(state_dict)

        CLIPVisionModelWithProjection wraps the vision transformer, so we load
        into its full state_dict via strict=False to accommodate any projection
        head differences, and fall back to the visual_model sub-module if needed.
        """
        print(f"[CLIPInterface] Loading un2CLIP weights from: {self.un2clip_ckpt_path}")
        state_dict = torch.load(
            self.un2clip_ckpt_path, map_location="cpu", weights_only=True
        )

        # Try loading directly into CLIPVisionModelWithProjection
        missing, unexpected = self.clip_image_encoder.load_state_dict(
            state_dict, strict=False
        )

        # If most keys are missing it means the ckpt uses the inner visual model
        # namespace — retry on the vision_model sub-module
        if len(missing) > len(state_dict) * 0.5:
            vm = getattr(self.clip_image_encoder, "vision_model", None)
            if vm is not None:
                missing, unexpected = vm.load_state_dict(state_dict, strict=False)

        if unexpected:
            print(
                f"[CLIPInterface] un2CLIP: {len(unexpected)} unexpected keys (ignored)."
            )
        if missing:
            print(
                f"[CLIPInterface] un2CLIP: {len(missing)} missing keys (kept from base CLIP)."
            )

        self.clip_image_encoder.eval()
        print("[CLIPInterface] un2CLIP visual encoder loaded successfully.")

    def getImgEmbedding(self, img: Image):
        # CLIP processor is usually callable; wrap defensively
        clip_inputs = self.clip_processor(images=img, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            clip_out = self.clip_image_encoder(**clip_inputs)
            if hasattr(clip_out, "image_embeds"):
                z_i = clip_out.image_embeds
            else:
                # fallback pool
                z_i = clip_out.last_hidden_state.mean(dim=1)
        return z_i

    def getTextEmbedding(self, text: Union[list[str], str]):
        # Text encoder is always standard CLIP regardless of use_un2clip,
        # since un2CLIP only fine-tunes the visual encoder.
        toks = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.text_encoder(**toks)
            if hasattr(out, "text_embeds"):
                return out.text_embeds
            else:
                return out.last_hidden_state.mean(dim=1)


class DiffusionInterface:

    def __init__(
        self,
        model_id="diffusers/stable-diffusion-2-1-unclip-i2i-l",
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        height=512,
        width=512,
        latent_scale=8,
        use_ddim: bool = False,
        ddim_eta: float = 0.0,
    ):
        """
        use_ddim: if True, replace the pipeline scheduler with DDIMScheduler
        ddim_eta: the eta parameter used by DDIM (0.0 = deterministic)
        """
        self.dtype = dtype
        self.seed = seed
        self.height = height
        self.width = width
        self.latent_scale = latent_scale
        self.device = device
        self.model_id = model_id
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta
        self.generator = None
        self.initDiffusionPipeline()
        self.getLatents()

    def getLatents(self):
        self.latents, self.generator = build_fixed_latent(
            self.pipe,
            self.height,
            self.width,
            self.device,
            latent_scale=self.latent_scale,
            seed=self.seed,
            dtype=self.dtype,
        )

    def initDiffusionPipeline(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype, trust_remote_code=True
        )
        # optionally switch to DDIM scheduler
        if self.use_ddim:
            try:
                # import locally to avoid requiring it when not used
                from diffusers import DDIMScheduler

                # construct a DDIM scheduler using the existing scheduler's config
                self.pipe.scheduler = DDIMScheduler.from_config(
                    self.pipe.scheduler.config
                )
            except Exception as e:
                # if DDIM isn't available or replacement fails, raise informative error
                raise RuntimeError(f"Failed to set DDIM scheduler: {e}")
        self.pipe = enable_memory_savings(self.pipe)
        self.pipe = self.pipe.to(self.device)

    def genImage(self, z, num_inference_steps=40, guidance_scale=8.0, latents=None):
        """
        Generate image(s) from CLIP embeddings (z) and fixed latents.
        If use_ddim=True the eta parameter will be passed to the pipeline call.
        """
        z = z.type(self.dtype).to(self.device)
        # ensure latents dtype/device
        if latents == None:
            latents = self.latents.to(device=self.device, dtype=self.dtype)
        else:
            latents = latents.to(device=self.device, dtype=self.dtype)
        # Build kwargs for the pipeline call
        call_kwargs = dict(
            image_embeds=z,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator,
        )
        if self.use_ddim:
            # DDIM supports 'eta' (controls stochasticity)
            call_kwargs["eta"] = float(self.ddim_eta)

        out = self.pipe(**call_kwargs)
        images = out.images if hasattr(out, "images") else out
        img = images[0] if isinstance(images, (list, tuple)) else images
        return img

    # ----- New helpers for encoding/decoding and DDIM inversion -----
    def pil_to_tensor(self, pil_image: Image, height: int, width: int, device, dtype):
        """Convert PIL image -> normalized tensor (1,3,H,W) suitable for VAE.encode."""
        pil_rgb = pil_image.convert("RGB").resize(
            (width, height), resample=Image.BICUBIC
        )
        arr = np.asarray(pil_rgb).astype(np.float32) / 255.0  # [0,1]
        # common diffusers VAE expects [-1,1]
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype)

    def encode_image_to_latents(self, pil_image: Image):
        """
        Robust encode -> latent using pipeline VAE without assuming image_processor is callable.
        Returns latents shaped (1,C_lat,H_lat,W_lat).
        """
        pipe = self.pipe
        device = self.device
        dtype = self.dtype

        # 1) Prefer pipeline's image_processor if it is callable and returns pixel_values
        proc = getattr(pipe, "image_processor", None)
        img_tensor = None
        if proc is not None and callable(proc):
            try:
                inputs = proc.preprocess(pil_image)
                if isinstance(inputs, dict) and "pixel_values" in inputs:
                    img_tensor = inputs["pixel_values"].to(device=device, dtype=dtype)
                elif isinstance(inputs, torch.Tensor):
                    img_tensor = inputs.to(device=device, dtype=dtype)
            except Exception:
                img_tensor = None

        # 2) fallback manual conversion
        if img_tensor is None:
            img_tensor = self.pil_to_tensor(
                pil_image, self.height, self.width, device, dtype
            )

        # 3) encode with VAE
        with torch.no_grad():
            enc_out = pipe.vae.encode(img_tensor)
            if hasattr(enc_out, "latent_dist"):
                ld = enc_out.latent_dist
                latents = ld.sample() if hasattr(ld, "sample") else ld
            elif isinstance(enc_out, (tuple, list)):
                latents = enc_out[0]
            else:
                latents = enc_out

        # Many diffusers multiply latents by a scaling factor after sampling; the decode expects the inverse.
        # If your pipeline uses a known scaling factor, adjust here as needed. We'll leave latents as returned.
        latents = latents * self.pipe.vae.config.scaling_factor
        return latents

    def decode_latents_to_image(self, latents):
        """Decode VAE latents back to a PIL image using pipeline VAE.decode."""
        pipe = self.pipe
        device = self.device
        dtype = self.dtype
        lat = latents.to(device=device, dtype=dtype)
        with torch.no_grad():
            dec = pipe.vae.decode(lat)
            # handle ModelOutput or tensor
            if hasattr(dec, "sample"):
                imgs = dec.sample
            elif isinstance(dec, (tuple, list)):
                imgs = dec[0]
            else:
                imgs = dec
        # expected [-1,1] -> [0,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        imgs = (imgs * 255).round().to(torch.uint8).cpu().numpy()
        # (B,C,H,W) -> HWC
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            arr = imgs[0].transpose(1, 2, 0)
        else:
            arr = imgs[0]
        return Image.fromarray(arr)

    def ddim_invert(self, pil_image: Image, num_inference_steps=40, noise_scale=1):
        """
        Perform a deterministic-ish DDIM inversion:
          - Encode image -> x0 (latents)
          - For each scheduler timestep, add deterministic noise using a reusable generator
          - Uses scheduler.add_noise(x0, noise, t) at each timestep to create a z_T
        Returns: z_T (tensor), timesteps (torch.Tensor)
        NOTE: This is a stable and deterministic approach using scheduler.add_noise and a single reusable generator.
        """
        # 1) Encode to latents (x_0 in latent space)
        x0 = self.encode_image_to_latents(pil_image)  # shape (1,C,H,W)
        # Ensure scheduler is DDIM-compatible (user should set use_ddim=True when init)
        scheduler = self.pipe.scheduler
        # set timesteps for inversion consistent with sampling
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps  # descending timesteps (tensor)

        # Create a single reusable deterministic generator (do not re-seed inside loop)
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)

        # We'll produce noise per-step deterministically and use scheduler.add_noise with x0.
        # This creates a deterministic z_T when the same seed & scheduler config are used.
        x_t = x0.clone().to(self.device, dtype=self.dtype)
        for i, t in enumerate(tqdm(list(timesteps))):
            # generate deterministic noise for this step
            noise = torch.randn(
                x_t.shape,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            )
            # add noise at timestep t based on x0 and this step's noise
            # (note: add_noise uses scheduler's formula to produce x_t from x0 and noise)
            x_t = scheduler.add_noise(x0, noise, t)
        # x_t is now the latent at final timestep (z_T)
        z_T = x_t
        return z_T, timesteps

    def reconstruct_from_zT(
        self,
        z_I,
        z_T,
        num_inference_steps=50,
        guidance_scale=1.0,
        eta=0.0,
        generator=None,
    ):
        """
        Run the pipeline from an initial latent z_T and return a PIL image.
        Requirements:
          - pipeline.scheduler should be DDIMScheduler with the same config used for inversion
          - use the same num_inference_steps used during inversion
          - use guidance_scale=1.0 for exact reconstruction
        """
        device = self.device
        # set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        out = self.pipe(
            image_embeds=z_I.to(self.dtype),
            latents=z_T.to(device=device, dtype=self.dtype),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            eta=float(eta),
        )
        images = out.images if hasattr(out, "images") else out
        img = images[0] if isinstance(images, (list, tuple)) else images

        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            img_tensor = img.detach().cpu()
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
            arr = (img_tensor * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(arr)
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        raise RuntimeError(f"Unhandled pipeline output type: {type(img)}")


class ConceptDrifter:

    def __init__(
        self,
        model_id="diffusers/stable-diffusion-2-1-unclip-i2i-l",
        clip_model="openai/clip-vit-large-patch14",
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        height=512,
        width=512,
        latent_scale=8,
        use_ddim: bool = False,
        ddim_eta: float = 0.0,
        vae_scaling_factor: float = 0.35,
        use_un2clip: bool = False,
        un2clip_ckpt_path: str | None = None,
    ):
        """
        Parameters
        ----------
        vae_scaling_factor  : Only applied when use_ddim=True. Overrides
                              pipe.vae.config.scaling_factor after pipeline load.
        use_un2clip         : If True, patch the CLIP visual encoder with
                              un2CLIP fine-tuned weights. The text encoder is
                              always kept as standard CLIP.
        un2clip_ckpt_path   : Path to the un2CLIP .ckpt file. Required when
                              use_un2clip=True (e.g.
                              './pretrained_models/openai_vit_l_14_224.ckpt').
        """
        self.device = device
        self.clip_interface = CLIPInterface(
            clip_model,
            dtype,
            device,
            use_un2clip=use_un2clip,
            un2clip_ckpt_path=un2clip_ckpt_path,
        )
        self.diff_interface = DiffusionInterface(
            model_id,
            dtype,
            device,
            seed,
            height,
            width,
            latent_scale,
            use_ddim,
            ddim_eta,
        )
        if use_ddim:
            self.diff_interface.pipe.vae.config.scaling_factor = vae_scaling_factor

    def getConceptVector(self, positive_text: str, negative_text: str):
        embeddings = self.clip_interface.getTextEmbedding(
            [positive_text, negative_text]
        )
        pos = embeddings[0:1]
        neg = embeddings[1:2]
        magnitude = (pos.norm(dim=1) + neg.norm(dim=1)) / 2
        return (pos - neg), magnitude

    def getProbs(self, z_i, z_pos, z_neg):
        prob_p = torch.nn.functional.cosine_similarity(z_i, z_pos)
        prob_n = torch.nn.functional.cosine_similarity(z_i, z_neg)
        probs = torch.nn.functional.softmax(torch.tensor([prob_p, prob_n]))
        return probs

    def perturbImagePoints(
        self,
        img,
        z_pos,
        z_neg,
        delta=0.1,
        latents=None,
        use_ddim: bool | None = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 8.0,
        ddim_guidance_scale: float = 32.0,
        ddim_eta: float = 0.0,
    ):
        """
        Perturb an image along a concept direction and return (base_img, perturbed_img).

        The perturbation is applied in CLIP embedding space:
          z_perturbed = z_i + delta * (z_pos - z_neg) * (||z_i|| / ||concept||)
        The direction is flipped automatically so the output always moves toward
        the negative concept (away from whatever the image already resembles most).

        Two reconstruction paths are supported:

        CLIP path (use_ddim=False):
          - Decode z_i and z_perturbed directly via the unCLIP pipeline using
            fixed random latents.  Fast but less faithful to the input image.

        DDIM path (use_ddim=True):
          - DDIM-invert the image to obtain a latent z_T that encodes its full
            pixel-level structure.
          - Reconstruct both base and perturbed images from the *same* z_T,
            only swapping the CLIP conditioning.  This anchors the output to the
            original image geometry while still shifting the concept.

        Parameters
        ----------
        img                  : PIL.Image  – source image (already resized to H×W).
        z_pos / z_neg        : torch.Tensor – CLIP text embeddings for the concept pair.
        delta                : float – perturbation strength.
        latents              : torch.Tensor | None – fixed latents for the CLIP path.
        use_ddim             : bool | None – True → DDIM path, False → CLIP path,
                               None → auto-detect from self.diff_interface.use_ddim.
        num_inference_steps  : int   – denoising steps for the CLIP path.
        guidance_scale       : float – guidance scale for the CLIP path.
        ddim_guidance_scale  : float – guidance scale for DDIM reconstruction
                               (higher values → stronger concept shift).
        ddim_eta             : float – DDIM eta (0.0 = fully deterministic).

        Returns
        -------
        (base_img, perturbed_img) : tuple[PIL.Image, PIL.Image]
        """
        # ── 1. Auto-detect DDIM mode ──────────────────────────────────────────
        if use_ddim is None:
            use_ddim = self.diff_interface.use_ddim

        # ── 2. Compute CLIP embedding and concept direction ───────────────────
        z_i = self.clip_interface.getImgEmbedding(img)
        z_i_mag = z_i.norm(dim=1)
        vector = z_pos - z_neg
        magnitude = (z_pos.norm(dim=1) + z_neg.norm(dim=1)) / 2

        # Flip direction so output always moves *away* from the closer concept
        probs = self.getProbs(z_i, z_pos, z_neg)
        if probs[0] > probs[1]:
            vector = -1 * vector

        z_perturbed = z_i + (delta * vector * (z_i_mag / magnitude))
        z_i = z_i.to(self.device)
        z_perturbed = z_perturbed.to(self.device)

        # ── 3a. CLIP path (original behaviour) ───────────────────────────────
        if not use_ddim:
            base_img = self.diff_interface.genImage(
                z_i,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=latents,
            )
            perturbed_img = self.diff_interface.genImage(
                z_perturbed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=latents,
            )

        # ── 3b. DDIM path ─────────────────────────────────────────────────────
        else:
            if not self.diff_interface.use_ddim:
                raise RuntimeError(
                    "DDIM path requested but DiffusionInterface was not initialised "
                    "with use_ddim=True. Re-create ConceptDrifter with use_ddim=True."
                )

            # Invert the image once → shared structural latent z_T
            z_T, _ = self.diff_interface.ddim_invert(
                img, num_inference_steps=num_inference_steps
            )

            # Fresh deterministic generator for reproducible decoding
            def _fresh_gen():
                g = torch.Generator(device=self.diff_interface.device)
                g.manual_seed(self.diff_interface.seed)
                return g

            # Base: reconstruct from z_T conditioned on the original CLIP embedding
            base_img = self.diff_interface.reconstruct_from_zT(
                z_i,
                z_T,
                num_inference_steps=num_inference_steps,
                guidance_scale=ddim_guidance_scale,
                eta=ddim_eta,
                generator=_fresh_gen(),
            )

            # Perturbed: same z_T, shifted CLIP conditioning
            perturbed_img = self.diff_interface.reconstruct_from_zT(
                z_perturbed,
                z_T,
                num_inference_steps=num_inference_steps,
                guidance_scale=ddim_guidance_scale,
                eta=ddim_eta,
                generator=_fresh_gen(),
            )

        # ── 4. Return in consistent (base, perturbed) order ───────────────────
        if probs[0] > probs[1]:
            # vector was flipped, so z_perturbed moves toward z_neg
            return perturbed_img, base_img
        else:
            return base_img, perturbed_img


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)

    image = Image.open("samples/test_img2.jpg").resize((512, 512))
    text_positive = "tiger"
    text_negative = "lion"

    # ── CLIP path (fast, fixed latents) ──────────────────────────────────────
    print("Running CLIP perturbation path ...")
    c_clip = ConceptDrifter(use_ddim=False)
    embeddings = c_clip.clip_interface.getTextEmbedding([text_positive, text_negative])
    z_pos = embeddings[0:1]
    z_neg = embeddings[1:2]

    base_clip, perturbed_clip = c_clip.perturbImagePoints(
        image,
        z_pos,
        z_neg,
        delta=0.2,
        use_ddim=False,  # explicit, but would also be auto-detected
        num_inference_steps=40,
        guidance_scale=8.0,
    )
    base_clip.save("outputs/clip_base.png")
    perturbed_clip.save("outputs/clip_perturbed.png")
    print("  Saved: outputs/clip_base.png, outputs/clip_perturbed.png")

    # ── DDIM path (structure-preserving, concept-shifted) ────────────────────
    print("Running DDIM perturbation path ...")
    c_ddim = ConceptDrifter(use_ddim=True, vae_scaling_factor=0.25)
    embeddings = c_ddim.clip_interface.getTextEmbedding([text_positive, text_negative])
    z_pos = embeddings[0:1]
    z_neg = embeddings[1:2]

    base_ddim, perturbed_ddim = c_ddim.perturbImagePoints(
        image,
        z_pos,
        z_neg,
        delta=0.5,
        use_ddim=True,  # explicit, but would also be auto-detected
        num_inference_steps=40,
        ddim_guidance_scale=32.0,
        ddim_eta=0.0,
    )
    base_ddim.save("outputs/ddim_base.png")
    perturbed_ddim.save("outputs/ddim_perturbed_2.png")
    print("  Saved: outputs/ddim_base.png, outputs/ddim_perturbed_2.png")

    # ── DDIM + un2CLIP path (detail-aware, structure-preserving) ─────────────
    print("Running DDIM + un2CLIP perturbation path ...")
    c_un2clip = ConceptDrifter(
        use_ddim=True,
        vae_scaling_factor=0.25,
        use_un2clip=True,
        un2clip_ckpt_path="./pretrained_models/openai_vit_l_14_224.ckpt",
    )
    embeddings = c_un2clip.clip_interface.getTextEmbedding(
        [text_positive, text_negative]
    )
    z_pos = embeddings[0:1]
    z_neg = embeddings[1:2]

    base_un2clip, perturbed_un2clip = c_un2clip.perturbImagePoints(
        image,
        z_pos,
        z_neg,
        delta=0.5,
        use_ddim=True,
        num_inference_steps=40,
        ddim_guidance_scale=32.0,
        ddim_eta=0.0,
    )
    base_un2clip.save("outputs/un2clip_base.png")
    perturbed_un2clip.save("outputs/un2clip_perturbed.png")
    print("  Saved: outputs/un2clip_base.png, outputs/un2clip_perturbed.png")
