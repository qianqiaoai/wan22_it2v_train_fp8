from contextlib import nullcontext
import inspect

import torch.distributed as dist


def _get_config(config, name, default):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


class FP8Context:
    def __init__(self, config):
        self.enabled = bool(_get_config(config, "fp8", False))
        self.recipe = None
        self.fp8_autocast = None
        self.fp8_group = None

        if not self.enabled:
            return

        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format
        except ImportError as exc:
            raise ImportError(
                "config.fp8=true requires transformer_engine. Install NVIDIA "
                "Transformer Engine in the training environment."
            ) from exc

        fp8_format = str(_get_config(config, "fp8_format", "HYBRID")).upper()
        fp8_format = getattr(Format, fp8_format)
        recipe_kwargs = {
            "fp8_format": fp8_format,
            "amax_history_len": int(_get_config(config, "fp8_amax_history_len", 16)),
            "amax_compute_algo": _get_config(config, "fp8_amax_compute_algo", "max"),
        }
        scaling_factor_compute_algo = _get_config(config, "fp8_scaling_factor_compute_algo", None)
        if scaling_factor_compute_algo is not None:
            recipe_kwargs["scaling_factor_compute_algo"] = scaling_factor_compute_algo
        override_linear_precision = _get_config(config, "fp8_override_linear_precision", None)
        if override_linear_precision is not None:
            recipe_kwargs["override_linear_precision"] = tuple(override_linear_precision)
        margin = _get_config(config, "fp8_margin", None)
        if margin is not None:
            recipe_kwargs["margin"] = int(margin)

        if "reduce_amax" in inspect.signature(DelayedScaling).parameters:
            recipe_kwargs["reduce_amax"] = bool(_get_config(config, "fp8_reduce_amax", True))
        for name in ("fp8_dpa", "fp8_mha"):
            if name in inspect.signature(DelayedScaling).parameters:
                recipe_kwargs[name] = bool(_get_config(config, name, False))

        self.recipe = DelayedScaling(**recipe_kwargs)
        self.fp8_autocast = te.fp8_autocast
        if bool(_get_config(config, "fp8_reduce_amax", True)) and dist.is_initialized():
            self.fp8_group = dist.group.WORLD

    def context(self):
        if not self.enabled:
            return nullcontext()

        kwargs = {"enabled": True}
        params = inspect.signature(self.fp8_autocast).parameters
        if "fp8_recipe" in params:
            kwargs["fp8_recipe"] = self.recipe
        elif "recipe" in params:
            kwargs["recipe"] = self.recipe
        if self.fp8_group is not None and "fp8_group" in params:
            kwargs["fp8_group"] = self.fp8_group

        return self.fp8_autocast(**kwargs)
