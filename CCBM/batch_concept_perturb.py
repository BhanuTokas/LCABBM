"""
batch_concept_perturb.py
------------------------
Applies concept perturbation (via ConceptDrifter) to multiple input folders,
each paired with its own set of concept pairs, and writes outputs into
per-folder sub-directories inside a root output folder.

Output structure
----------------
output_dir/
  <folder_name>/
    image_01_base.png
    image_01_<concept>_delta0.1.png
    image_01_<concept>_delta0.2.png
    ...
  <folder_name_2>/
    ...
  run_summary.json

Behaviour
---------
- Each input_dir is processed independently with its own concept pairs.
- Only images directly inside each input_dir are processed (no recursion).
- Skips any output file that already exists (resume-friendly).
- Skips and logs any image that fails; partial failures are also logged.
- Saves a run_summary.json to output_dir on completion.

Config file format (concepts.json)
------------------------------------
[
  {
    "input_dir": "path/to/animals",
    "output_name": "animals",           <- sub-folder name inside output_dir
    "concepts": [
      { "name": "tiger_vs_lion", "positive": "tiger", "negative": "lion" },
      { "name": "day_vs_night",  "positive": "daytime", "negative": "nighttime" }
    ]
  },
  {
    "input_dir": "path/to/landscapes",
    "output_name": "landscapes",
    "concepts": [
      { "name": "summer_vs_winter", "positive": "summer", "negative": "winter" }
    ]
  }
]

Usage
-----
python batch_concept_perturb.py \\
    --output_dir  path/to/output \\
    --config      concepts.json \\
    --deltas      0.1 0.2 0.5 1.0 \\
    [--seed 42] \\
    [--max_images 20] \\
    [--height 512] \\
    [--width 512] \\
    [--num_inference_steps 40] \\
    [--guidance_scale 8.0] \\
    [--use_ddim] \\
    [--ddim_guidance_scale 32.0] \\
    [--ddim_eta 0.0] \\
    [--vae_scaling_factor 0.35] \\
    [--use_un2clip] \\
    [--un2clip_ckpt_path ./pretrained_models/openai_vit_l_14_224.ckpt] \\
    [--dtype float16] \\
    [--model_id diffusers/stable-diffusion-2-1-unclip-i2i-l] \\
    [--clip_model openai/clip-vit-large-patch14]
"""

import argparse
import json
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ConceptGenDDIM import ConceptDrifter  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> list[dict]:
    """
    Load and validate the JSON config.
    Each entry must have: input_dir, output_name, concepts (list of name/positive/negative).
    """
    with open(config_path, "r") as f:
        entries = json.load(f)

    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError("concepts.json must be a non-empty JSON array.")

    for i, entry in enumerate(entries):
        for key in ("input_dir", "output_name", "concepts"):
            if key not in entry:
                raise ValueError(f"Entry {i} is missing the '{key}' field.")
        if not isinstance(entry["concepts"], list) or len(entry["concepts"]) == 0:
            raise ValueError(f"Entry {i} 'concepts' must be a non-empty list.")
        for j, c in enumerate(entry["concepts"]):
            for key in ("name", "positive", "negative"):
                if key not in c:
                    raise ValueError(
                        f"Entry {i}, concept {j} is missing the '{key}' field."
                    )

    return entries


def collect_images(input_dir: Path) -> list[Path]:
    """Return all image files (sorted) found directly inside input_dir (no recursion)."""
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_stem(path: Path) -> str:
    """Return a filename-safe version of the image stem."""
    return path.stem.replace(" ", "_")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_image(
    img_path: Path,
    drifter: ConceptDrifter,
    concepts: list[dict],
    deltas: list[float],
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    ddim_guidance_scale: float,
    ddim_eta: float,
) -> dict:
    """
    Process a single image:
      1. Load & resize.
      2. For each concept pair x delta, call perturbImagePoints which returns
         (base_img, perturbed_img).  The base is saved once per image (first
         concept/delta that produces it); subsequent calls skip it if it already
         exists.
      3. All outputs are skipped if they already exist (resume-friendly).

    When use_ddim=True (auto-detected from the drifter), perturbImagePoints
    inverts the image via DDIM and reconstructs both base and perturbed from
    the same z_T, preserving image structure while shifting the concept.
    """
    stem = safe_stem(img_path)
    result = {
        "image": img_path.name,
        "status": "success",
        "outputs": [],
        "skipped_existing": [],
        "errors": [],
    }

    # Load and resize
    img = (
        Image.open(img_path)
        .convert("RGB")
        .resize((drifter.diff_interface.width, drifter.diff_interface.height))
    )

    base_path = output_dir / f"{stem}_base.png"
    base_saved = base_path.exists()  # track whether base has been saved already

    if base_saved:
        print(f"  [Base] Already exists, skipping -> {base_path.name}")
        result["skipped_existing"].append(base_path.name)

    # Concept x delta sweep
    for concept in concepts:
        cname = concept["name"]
        pos_text = concept["positive"]
        neg_text = concept["negative"]

        embeddings = drifter.clip_interface.getTextEmbedding([pos_text, neg_text])
        z_pos = embeddings[0:1]
        z_neg = embeddings[1:2]

        for delta in deltas:
            out_name = f"{stem}_{cname}_delta{delta}.png"
            out_path = output_dir / out_name

            # Both outputs exist — nothing to do
            if out_path.exists() and base_saved:
                print(f"  [Perturb] Already exists, skipping -> {out_name}")
                result["skipped_existing"].append(out_name)
                continue

            try:
                print(f"  [Perturb] concept='{cname}'  delta={delta} ...")
                base_img, perturbed_img = drifter.perturbImagePoints(
                    img,
                    z_pos,
                    z_neg,
                    delta=delta,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    ddim_guidance_scale=ddim_guidance_scale,
                    ddim_eta=ddim_eta,
                )

                # Save base once (first time it is produced)
                if not base_saved:
                    base_img.save(base_path)
                    result["outputs"].append(base_path.name)
                    print(f"  [Base]   Saved -> {base_path.name}")
                    base_saved = True

                if out_path.exists():
                    print(f"  [Perturb] Already exists, skipping -> {out_name}")
                    result["skipped_existing"].append(out_name)
                else:
                    perturbed_img.save(out_path)
                    result["outputs"].append(out_name)
                    print(f"  [Perturb] Saved -> {out_name}")

            except Exception as e:
                msg = f"concept='{cname}' delta={delta}: {e}"
                print(f"  [WARNING] Skipping failed perturbation — {msg}")
                traceback.print_exc()
                result["errors"].append(msg)

    if result["errors"]:
        result["status"] = "partial"

    return result


def process_entry(
    entry: dict,
    drifter: ConceptDrifter,
    root_output_dir: Path,
    deltas: list[float],
    seed: int,
    max_images: int | None,
    num_inference_steps: int,
    guidance_scale: float,
    ddim_guidance_scale: float,
    ddim_eta: float,
) -> dict:
    """
    Process one config entry (one input_dir with its own concept pairs).
    Returns a summary dict for this entry.
    """
    input_dir = Path(entry["input_dir"])
    output_name = entry["output_name"]
    concepts = entry["concepts"]
    output_dir = root_output_dir / output_name

    print(f"\n{'='*60}")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Concepts: {[c['name'] for c in concepts]}")
    print(f"{'='*60}")

    entry_summary = {
        "input_dir": str(input_dir),
        "output_name": output_name,
        "concepts": concepts,
        "total_available": 0,
        "processed": 0,
        "succeeded": 0,
        "skipped": 0,
        "results": [],
        "skipped_details": [],
    }

    if not input_dir.is_dir():
        msg = f"input_dir does not exist: {input_dir}"
        print(f"  [ERROR] {msg} — skipping this entry.")
        entry_summary["skipped_details"].append({"entry": output_name, "error": msg})
        return entry_summary

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect, shuffle, cap
    images = collect_images(input_dir)
    if not images:
        print(f"  [WARNING] No images found in '{input_dir}' — skipping.")
        return entry_summary

    random.seed(seed)
    random.shuffle(images)

    total_available = len(images)
    entry_summary["total_available"] = total_available

    if max_images is not None and max_images < total_available:
        images = images[:max_images]
        print(
            f"Found {total_available} image(s); randomly selected {len(images)} "
            f"(--max_images={max_images}, seed={seed})."
        )
    else:
        print(f"Found {total_available} image(s) (processing all).")

    entry_summary["processed"] = len(images)
    results = []
    skipped = []

    for idx, img_path in enumerate(images, start=1):
        print(f"\n  [{idx}/{len(images)}] {img_path.name}")
        try:
            result = process_image(
                img_path=img_path,
                drifter=drifter,
                concepts=concepts,
                deltas=deltas,
                output_dir=output_dir,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ddim_guidance_scale=ddim_guidance_scale,
                ddim_eta=ddim_eta,
            )
            results.append(result)
            if result["status"] == "partial":
                print(f"  [WARNING] '{img_path.name}' finished with partial errors.")
        except Exception as e:
            msg = str(e)
            print(f"\n  [SKIPPED] '{img_path.name}' failed entirely: {msg}")
            traceback.print_exc()
            skipped.append({"image": img_path.name, "error": msg})

    entry_summary["succeeded"] = len(results)
    entry_summary["skipped"] = len(skipped)
    entry_summary["results"] = results
    entry_summary["skipped_details"] = skipped

    return entry_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch concept perturbation across multiple input folders."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output folder. Each input_dir gets its own sub-folder here.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to concepts.json mapping input_dirs to their concept pairs.",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.5, 1.0],
        help="List of delta perturbation strengths (default: 0.1 0.2 0.5 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        help="Guidance scale for the CLIP (non-DDIM) path (default: 8.0).",
    )
    parser.add_argument(
        "--ddim_guidance_scale",
        type=float,
        default=32.0,
        help="Guidance scale for the DDIM reconstruction path (default: 32.0).",
    )
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Max images to process per input_dir (random shuffle before cap). "
        "Omit to process all images.",
    )
    parser.add_argument(
        "--use_ddim",
        action="store_true",
        help="Replace the default scheduler with DDIMScheduler.",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="Eta for DDIM (0.0 = deterministic). Only used with --use_ddim.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--model_id",
        default="diffusers/stable-diffusion-2-1-unclip-i2i-l",
    )
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--vae_scaling_factor",
        type=float,
        default=0.35,
        help="VAE scaling factor override applied when --use_ddim is set (default: 0.35).",
    )
    parser.add_argument(
        "--use_un2clip",
        action="store_true",
        help="Patch the CLIP visual encoder with un2CLIP fine-tuned weights. "
        "Requires --un2clip_ckpt_path. Text encoder remains standard CLIP.",
    )
    parser.add_argument(
        "--un2clip_ckpt_path",
        type=str,
        default=None,
        help="Path to the un2CLIP checkpoint file "
        "(e.g. ./pretrained_models/openai_vit_l_14_224.ckpt). "
        "Required when --use_un2clip is set.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    root_output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    if not config_path.is_file():
        sys.exit(f"[ERROR] Config file not found: {config_path}")

    root_output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Load config
    print(f"Loading config: {config_path}")
    entries = load_config(config_path)
    print(f"  {len(entries)} input folder(s) defined.")
    print(f"  Delta sweep : {args.deltas}")
    if args.max_images:
        print(f"  Max images  : {args.max_images} per folder (seed={args.seed})")

    # Validate un2clip args early
    if args.use_un2clip and not args.un2clip_ckpt_path:
        sys.exit("[ERROR] --un2clip_ckpt_path is required when --use_un2clip is set.")
    if args.use_un2clip and not Path(args.un2clip_ckpt_path).is_file():
        sys.exit(f"[ERROR] un2CLIP checkpoint not found: {args.un2clip_ckpt_path}")

    # Initialise ConceptDrifter once (shared across all entries)
    print("\nInitialising ConceptDrifter (loading models — this may take a moment) ...")
    if args.use_un2clip:
        print(f"  un2CLIP visual encoder will be loaded from: {args.un2clip_ckpt_path}")
    drifter = ConceptDrifter(
        model_id=args.model_id,
        clip_model=args.clip_model,
        dtype=dtype,
        seed=args.seed,
        height=args.height,
        width=args.width,
        use_ddim=args.use_ddim,
        ddim_eta=args.ddim_eta,
        vae_scaling_factor=args.vae_scaling_factor,
        use_un2clip=args.use_un2clip,
        un2clip_ckpt_path=args.un2clip_ckpt_path,
    )
    print("ConceptDrifter ready.")

    # Process each entry
    run_start = datetime.now().isoformat()
    entry_summaries = []

    for entry in entries:
        summary = process_entry(
            entry=entry,
            drifter=drifter,
            root_output_dir=root_output_dir,
            deltas=args.deltas,
            seed=args.seed,
            max_images=args.max_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ddim_guidance_scale=args.ddim_guidance_scale,
            ddim_eta=args.ddim_eta,
        )
        entry_summaries.append(summary)

    # Save run_summary.json
    total_succeeded = sum(e["succeeded"] for e in entry_summaries)
    total_skipped = sum(e["skipped"] for e in entry_summaries)

    summary = {
        "run_start": run_start,
        "run_end": datetime.now().isoformat(),
        "settings": {
            "output_dir": str(root_output_dir),
            "config": str(config_path),
            "deltas": args.deltas,
            "seed": args.seed,
            "max_images": args.max_images,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "ddim_guidance_scale": args.ddim_guidance_scale,
            "use_ddim": args.use_ddim,
            "ddim_eta": args.ddim_eta,
            "vae_scaling_factor": args.vae_scaling_factor,
            "use_un2clip": args.use_un2clip,
            "un2clip_ckpt_path": args.un2clip_ckpt_path,
            "dtype": args.dtype,
            "model_id": args.model_id,
            "clip_model": args.clip_model,
        },
        "total_folders": len(entries),
        "total_succeeded": total_succeeded,
        "total_skipped": total_skipped,
        "entries": entry_summaries,
    }

    summary_path = root_output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final report
    print("\n" + "=" * 60)
    print("Batch complete.")
    print(f"  Folders processed : {len(entries)}")
    print(f"  Images succeeded  : {total_succeeded}")
    print(f"  Images skipped    : {total_skipped}")
    print(f"  Output root       : {root_output_dir}")
    print(f"  Summary log       : {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
