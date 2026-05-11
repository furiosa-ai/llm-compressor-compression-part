import functools
from collections import namedtuple

from loguru import logger

__all__ = [
    "get_layer_mappings_from_architecture",
    "get_nonfused_smooth_layers",
    "MAPPINGS_REGISTRY",
    "NONFUSED_SMOOTH_REGISTRY",
    "DEFAULT_SMOOTHQUANT_MAPPINGS",
    "DEFAULT_NONFUSED_SMOOTH_LAYERS",
]

LayerMapType = tuple[list[str], str]
LayerMap: LayerMapType = namedtuple("LayerMap", ["balance_layers", "smooth_layers"])

DEFAULT_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_proj", "re:.*up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
MIXTRAL_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
]
BLOOM_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*query_key_value"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*dense_h_to_4h"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
PHI3_VISION_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*qkv_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
WHISPER_V2_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*k_proj", "re:.*v_proj", "re:.*q_proj"],
        smooth_layers="re:.*self_attn_layer_norm",
    ),
    LayerMap(
        balance_layers=["re:.*fc1"],
        smooth_layers="re:.*final_layer_norm",
    ),
]

DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q(_a)?_proj$", "re:.*kv_a_proj_with_mqa"],
        smooth_layers="re:.*input_layernorm",
    ),
]

AFMOE_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=[
            "re:.*self_attn\\.q_proj",
            "re:.*self_attn\\.k_proj",
            "re:.*self_attn\\.v_proj",
            "re:.*self_attn\\.gate_proj",
        ],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*mlp.*gate_proj", "re:.*mlp.*up_proj"],
        smooth_layers="re:.*pre_mlp_layernorm",
    ),
]


# Linear layers to self-smooth ("non-fused"): no preceding LayerNorm to absorb
# the migration scale into. Calibration captures their INPUT activations and
# applies migration via a per-input-channel divisor stored as a persistent
# `smooth_scale` buffer on the module; compressed-tensors' quantized_forward
# divides the input by it before activation quantization. Mirrors the
# `_smooth_linear_output` path in INT_vs_FP/quant/smoothquant.py.
DEFAULT_NONFUSED_SMOOTH_LAYERS: list[str] = [
    "re:.*self_attn\\.o_proj$",
    "re:.*mlp\\.down_proj$",
]

NONFUSED_SMOOTH_REGISTRY: dict[str, list[str]] = {
    "Gemma2ForCausalLM": DEFAULT_NONFUSED_SMOOTH_LAYERS,
    "Gemma3ForCausalLM": DEFAULT_NONFUSED_SMOOTH_LAYERS,
    "LlamaForCausalLM": DEFAULT_NONFUSED_SMOOTH_LAYERS,
    "MistralForCausalLM": DEFAULT_NONFUSED_SMOOTH_LAYERS,
    "Qwen2ForCausalLM": DEFAULT_NONFUSED_SMOOTH_LAYERS,
    "Qwen3ForCausalLM": DEFAULT_NONFUSED_SMOOTH_LAYERS,
}


def get_nonfused_smooth_layers(architecture: str) -> list[str]:
    """Per-architecture list of regexes for Linear layers to self-smooth
    (no preceding LayerNorm). Returns [] when architecture is unknown."""
    return NONFUSED_SMOOTH_REGISTRY.get(architecture, [])


# Registry of layer mappings for different architectures
#   Add more mappings here
MAPPINGS_REGISTRY: dict[str, list[LayerMap]] = {
    "BloomForCausalLM": BLOOM_SMOOTHQUANT_MAPPINGS,
    "ChatGLMForConditionalGeneration": BLOOM_SMOOTHQUANT_MAPPINGS,
    "DeepseekV2ForCausalLM": DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
    "Gemma2ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Gemma3ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Gemma3ForConditionalGeneration": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Glm4MoeForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "GlmMoeDsaForCausalLM": DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
    "Llama4ForConditionalGeneration": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "LlamaForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Mistral3ForConditionalGeneration": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "MistralForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "MixtralForCausalLM": MIXTRAL_SMOOTHQUANT_MAPPINGS,
    "Phi3VForCausalLM": PHI3_VISION_SMOOTHQUANT_MAPPINGS,
    "Qwen2ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Qwen3ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "WhisperForConditionalGeneration": WHISPER_V2_SMOOTHQUANT_MAPPINGS,
    "AfmoeForCausalLM": AFMOE_SMOOTHQUANT_MAPPINGS,
}


def get_layer_mappings_from_architecture(architecture: str) -> list[LayerMap]:
    """
    :param architecture: str: The architecture of the model
    :return: list: The layer mappings for the given architecture
    """

    if architecture not in MAPPINGS_REGISTRY:
        logger.info(
            f"Architecture {architecture} not found in mappings. "
            f"Using default mappings: {DEFAULT_SMOOTHQUANT_MAPPINGS}"
        )

    return MAPPINGS_REGISTRY.get(architecture, DEFAULT_SMOOTHQUANT_MAPPINGS)


def handle_mapping_resolution_errors(func):
    """
    Decorator to catch any errors that occur when resolving mappings and provide a
    helpful error message to the user pointing them to the README
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as original_exception:
            readme_location = (
                "https://github.com/vllm-project/llm-compressor/tree/main/"
                "src/llmcompressor/modifiers/transform/smoothquant"
            )
            raise RuntimeError(
                f"Error resolving mappings for given architecture."
                f"Please refer to the README at {readme_location} for more information."
            ) from original_exception

    return wrapper
