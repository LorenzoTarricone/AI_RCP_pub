"""One-time setup helpers run on demand before model loading."""

import os
import glob
import logging

logger = logging.getLogger(__name__)


def ensure_chemberta_safetensors(
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
) -> None:
    """Convert the cached ChemBERTa weights from ``pytorch_model.bin`` to
    ``model.safetensors`` if the safetensors file is missing.

    Why this exists: ``seyonec/ChemBERTa-zinc-base-v1`` only ships a legacy
    ``pytorch_model.bin``. Since transformers >= 4.46 (CVE-2025-32434),
    ``AutoModel.from_pretrained`` refuses to call ``torch.load`` when torch
    is older than 2.6, so the bin path fails on osx-arm64 (where the only
    conda-forge dgl build pins torch to 2.3). Pre-staging a safetensors
    copy in the HuggingFace cache lets transformers take the safetensors
    code path and skip ``torch.load`` entirely.

    The function is a no-op once the conversion has been done.
    """
    try:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import save_file
    except ImportError as e:
        logger.debug("ensure_chemberta_safetensors: skipping (%s)", e)
        return

    try:
        local_dir = snapshot_download(model_name)
    except Exception as e:
        logger.debug("ensure_chemberta_safetensors: snapshot_download failed (%s)", e)
        return

    bin_path = os.path.join(local_dir, "pytorch_model.bin")
    sft_path = os.path.join(local_dir, "model.safetensors")

    if os.path.exists(sft_path) or not os.path.exists(bin_path):
        return

    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    state = {k: v.contiguous().clone() for k, v in state.items()}
    save_file(state, sft_path)
    logger.info("Converted %s to safetensors at %s", bin_path, sft_path)

    no_exist_root = os.path.join(
        os.path.dirname(os.path.dirname(local_dir)), ".no_exist"
    )
    for marker in glob.glob(os.path.join(no_exist_root, "*", "model.safetensors")):
        try:
            os.remove(marker)
        except OSError:
            pass
