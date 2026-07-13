"""
LoRA Update Runtime - Manages plus/minus LoRA slots for ZO perturbation.

Uses fixed vLLM LoRA slots and training-only direct slot updates.
Supports all 2D trainable parameters (Linear layers).
"""

from functools import partial
from typing import Any, Dict, List, Sequence
import torch

from zo_vllm.config import DEFAULT_ZO_MINUS_LORA_ID, DEFAULT_ZO_PLUS_LORA_ID
from zo_vllm.core.lora_runtime.peft_tensors import peft_tensor_base_shapes
from zo_vllm.core.lora_runtime.slot_validation import (
    validate_direct_lora_slot_structure,
)
from zo_vllm.core.lora_runtime.slot_writer import (
    _update_lora_slot_from_direction_in_vllm_model,
    _update_lora_slots_from_directions_in_vllm_model,
    _update_lora_slots_in_vllm_model,
)
from zo_vllm.core.lora_scope import DEFAULT_TRANSFORMER_TARGET_MODULES
from zo_vllm.core.weight_sync import PACKED_MODEL_LAYERS_MODEL_TYPES

DIRECT_SLOT_PATH_PREFIX = "/zo_vllm_direct_lora"
SUPPORTED_CONFIG_MODEL_TYPES = {"opt", *PACKED_MODEL_LAYERS_MODEL_TYPES}


def _build_lora_model_from_tensors(
    manager,
    *,
    lora_id: int,
    config: dict,
    tensors: Dict[str, torch.Tensor],
):
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    peft_config = dict(config)
    peft_config.setdefault("vllm_max_position_embeddings", None)
    peft_helper = PEFTHelper.from_dict(peft_config)
    peft_helper.validate_legal(manager.lora_config)

    hf_to_vllm_mapper = getattr(manager.model, "hf_to_vllm_mapper", None)
    lora_skip_prefixes = getattr(manager.model, "lora_skip_prefixes", None)
    return LoRAModel.from_lora_tensors(
        lora_model_id=lora_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device=str(manager.device),
        dtype=manager.lora_config.lora_dtype,
        model_vocab_size=manager.vocab_size,
        weights_mapper=hf_to_vllm_mapper,
        skip_prefixes=lora_skip_prefixes,
    )


def _load_two_lora_tensors_into_vllm_model(
    model,
    *,
    plus_id: int,
    minus_id: int,
    config: dict,
    plus_tensors: Dict[str, torch.Tensor],
    minus_tensors: Dict[str, torch.Tensor],
) -> dict:
    """Create and activate both persistent direct-training LoRA slots."""
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError(
            "vLLM model has no lora_manager; enable_lora=True is required"
        )
    plus_structure = validate_direct_lora_slot_structure(
        manager,
        direction_shapes=peft_tensor_base_shapes(plus_tensors),
        source="direct_lora_plus_tensors",
    )
    minus_structure = validate_direct_lora_slot_structure(
        manager,
        direction_shapes=peft_tensor_base_shapes(minus_tensors),
        source="direct_lora_minus_tensors",
    )

    plus_lora = _build_lora_model_from_tensors(
        manager,
        lora_id=plus_id,
        config=config,
        tensors=plus_tensors,
    )
    minus_lora = _build_lora_model_from_tensors(
        manager,
        lora_id=minus_id,
        config=dict(config),
        tensors=minus_tensors,
    )

    plus_removed = manager.remove_adapter(plus_lora.id)
    minus_removed = manager.remove_adapter(minus_lora.id)
    plus_loaded = manager.add_adapter(plus_lora)
    minus_loaded = manager.add_adapter(minus_lora)
    plus_activated = manager.activate_adapter(plus_lora.id)
    minus_activated = manager.activate_adapter(minus_lora.id)
    return {
        "plus": {
            "lora_id": int(plus_lora.id),
            "removed_existing": bool(plus_removed),
            "loaded": bool(plus_loaded),
            "activated": bool(plus_activated),
            "device": str(manager.device),
            "num_tensors": len(plus_tensors),
            "structure_info": plus_structure,
        },
        "minus": {
            "lora_id": int(minus_lora.id),
            "removed_existing": bool(minus_removed),
            "loaded": bool(minus_loaded),
            "activated": bool(minus_activated),
            "device": str(manager.device),
            "num_tensors": len(minus_tensors),
            "structure_info": minus_structure,
        },
    }


def _load_empty_lora_slots_from_runtime_config(
    model,
    *,
    runtime_config: dict[str, Any],
) -> dict:
    """Create persistent empty LoRA slots inside the worker process.

    Serving-time ZO can use very large LoRA ranks. Sending the empty adapter
    tensors through the engine RPC can exceed msgpack limits, so the worker
    builds the registration tensors locally from compact model metadata.
    """

    runtime = LoRAUpdateRuntime(**runtime_config)
    runtime._set_ids_for_step(step=0)
    config = runtime._build_config()
    empty_tensors = runtime._build_empty_tensors()
    return _load_two_lora_tensors_into_vllm_model(
        model,
        plus_id=runtime.plus_id,
        minus_id=runtime.minus_id,
        config=config,
        plus_tensors=empty_tensors,
        minus_tensors=empty_tensors.copy(),
    )


def _remove_lora_from_vllm_model(model, *, lora_id: int) -> bool:
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        return False
    return bool(manager.remove_adapter(lora_id))


class LoRAUpdateRuntime:
    """
    Manages two direct vLLM LoRA slots for ZO plus/minus perturbation.

    Slots:
    - plus_id: Perturbation +eps*U @ V^T
    - minus_id: Perturbation -eps*U @ V^T

    Supports all 2D trainable parameters (Linear layers).
    """

    def __init__(
        self,
        rank: int,
        num_layers: int = 32,
        plus_id: int = DEFAULT_ZO_PLUS_LORA_ID,
        minus_id: int = DEFAULT_ZO_MINUS_LORA_ID,
        llm: Any | None = None,
        base_model_name: str = "facebook/opt-2.7b",
        hidden_size: int = 2560,
        ffn_dim: int = 10240,
        target_modules: Sequence[str] | None = None,
        model_type: str = "opt",
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        vocab_size: int | None = None,
        embedding_dim: int | None = None,
        tie_word_embeddings: bool = False,
    ):
        self.rank = rank
        self.num_layers = num_layers
        self.plus_id = plus_id
        self.minus_id = minus_id
        self._base_plus_id = plus_id
        self._base_minus_id = minus_id
        self.llm = llm
        self.base_model_name = base_model_name
        self.hidden_size = int(hidden_size)
        self.ffn_dim = int(ffn_dim)
        self.model_type = str(model_type)
        self.num_attention_heads = int(
            self.hidden_size if num_attention_heads is None else num_attention_heads
        )
        self.num_key_value_heads = int(
            self.num_attention_heads
            if num_key_value_heads is None
            else num_key_value_heads
        )
        self.head_dim = None if head_dim is None else int(head_dim)
        self.vocab_size = None if vocab_size is None else int(vocab_size)
        self.embedding_dim = (
            self.hidden_size if embedding_dim is None else int(embedding_dim)
        )
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.target_modules = list(target_modules or DEFAULT_TRANSFORMER_TARGET_MODULES)
        self.plus_name = "lozo_plus"
        self.minus_name = "lozo_minus"
        self.last_update_info: dict[str, Any] = {}

        self.plus_path: str = ""
        self.minus_path: str = ""
        self._registered = False

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        rank: int,
        base_model_name: str,
        **kwargs: Any,
    ) -> "LoRAUpdateRuntime":
        """Build a runtime from a Hugging Face model config."""
        target_modules = kwargs.get("target_modules")
        model_type = str(getattr(model_config, "model_type", ""))
        if target_modules is None and model_type not in SUPPORTED_CONFIG_MODEL_TYPES:
            raise ValueError(
                "LoRAUpdateRuntime.from_model_config currently supports only "
                f"{sorted(SUPPORTED_CONFIG_MODEL_TYPES)} module layouts when "
                f"target_modules is not provided; got model_type={model_type!r}. "
                "Add an explicit model mapping before using this model."
            )
        num_layers = getattr(model_config, "num_hidden_layers")
        hidden_size = getattr(model_config, "hidden_size")
        ffn_dim = getattr(
            model_config,
            "ffn_dim",
            getattr(model_config, "intermediate_size", None),
        )
        if ffn_dim is None:
            raise ValueError("model_config must define ffn_dim or intermediate_size")
        return cls(
            rank=rank,
            num_layers=int(num_layers),
            base_model_name=base_model_name,
            hidden_size=int(hidden_size),
            ffn_dim=int(ffn_dim),
            model_type=model_type,
            num_attention_heads=int(
                getattr(model_config, "num_attention_heads", hidden_size)
            ),
            num_key_value_heads=int(
                getattr(
                    model_config,
                    "num_key_value_heads",
                    getattr(model_config, "num_attention_heads", hidden_size),
                )
            ),
            head_dim=getattr(model_config, "head_dim", None),
            vocab_size=getattr(model_config, "vocab_size", None),
            embedding_dim=int(
                getattr(model_config, "word_embed_proj_dim", hidden_size)
            ),
            tie_word_embeddings=bool(
                getattr(model_config, "tie_word_embeddings", False)
            ),
            **kwargs,
        )

    @property
    def request_load_inplace(self) -> bool:
        """Whether vLLM should reload tensors from the request path."""
        return False

    def attach_llm(self, llm: Any) -> None:
        self.llm = llm

    def _request_path(self, lora_id: int) -> str:
        return f"{DIRECT_SLOT_PATH_PREFIX}/{lora_id}"

    def worker_registration_config(self) -> dict[str, Any]:
        """Return compact metadata for worker-local empty slot registration."""

        return {
            "rank": int(self.rank),
            "num_layers": int(self.num_layers),
            "plus_id": int(self.plus_id),
            "minus_id": int(self.minus_id),
            "llm": None,
            "base_model_name": self.base_model_name,
            "hidden_size": int(self.hidden_size),
            "ffn_dim": int(self.ffn_dim),
            "target_modules": list(self.target_modules),
            "model_type": self.model_type,
            "num_attention_heads": int(self.num_attention_heads),
            "num_key_value_heads": int(self.num_key_value_heads),
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "embedding_dim": int(self.embedding_dim),
            "tie_word_embeddings": bool(self.tie_word_embeddings),
        }

    def _build_target_modules(self) -> List[str]:
        """
        Build list of target modules for LoRA.

        Returns short names (vLLM expects this format).
        """
        return list(self.target_modules)

    def _build_empty_tensors(self) -> Dict[str, torch.Tensor]:
        """Build empty tensors for registration (PEFT format, only Linear modules)."""
        tensors = {}
        target_modules = set(self._build_target_modules())

        if self.model_type in PACKED_MODEL_LAYERS_MODEL_TYPES:
            head_dim = int(
                self.head_dim or (self.hidden_size // self.num_attention_heads)
            )
            q_size = self.num_attention_heads * head_dim
            kv_size = self.num_key_value_heads * head_dim
            module_specs = [
                ("self_attn.q_proj", q_size, self.hidden_size),
                ("self_attn.k_proj", kv_size, self.hidden_size),
                ("self_attn.v_proj", kv_size, self.hidden_size),
                ("self_attn.o_proj", self.hidden_size, q_size),
                ("mlp.gate_proj", self.ffn_dim, self.hidden_size),
                ("mlp.up_proj", self.ffn_dim, self.hidden_size),
                ("mlp.down_proj", self.hidden_size, self.ffn_dim),
            ]
            for layer_idx in range(self.num_layers):
                for suffix, out_features, in_features in module_specs:
                    if not self._target_enabled(suffix.rsplit(".", 1)[-1]):
                        continue
                    module_path = f"model.layers.{layer_idx}.{suffix}"
                    lora_A = torch.zeros(self.rank, in_features, dtype=torch.float16)
                    lora_B = torch.zeros(out_features, self.rank, dtype=torch.float16)
                    tensors[f"base_model.model.{module_path}.lora_A.weight"] = lora_A
                    tensors[f"base_model.model.{module_path}.lora_B.weight"] = lora_B
            self._add_embedding_tensors(tensors, embed_module_path="model.embed_tokens")
            self._add_lm_head_tensors(tensors)
            return tensors

        linear_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

        for layer_idx in range(self.num_layers):
            for module_name in linear_modules:
                if module_name not in target_modules:
                    continue
                if module_name in ["fc1", "fc2"]:
                    module_path = f"model.decoder.layers.{layer_idx}.{module_name}"
                else:
                    module_path = (
                        f"model.decoder.layers.{layer_idx}.self_attn.{module_name}"
                    )

                # Determine dimensions
                if module_name == "fc1":
                    out_features, in_features = self.ffn_dim, self.hidden_size
                elif module_name == "fc2":
                    out_features, in_features = self.hidden_size, self.ffn_dim
                else:
                    out_features, in_features = self.hidden_size, self.hidden_size

                lora_A = torch.zeros(self.rank, in_features, dtype=torch.float16)
                lora_B = torch.zeros(out_features, self.rank, dtype=torch.float16)

                # PEFT format keys
                tensors[f"base_model.model.{module_path}.lora_A.weight"] = lora_A
                tensors[f"base_model.model.{module_path}.lora_B.weight"] = lora_B

        self._add_embedding_tensors(
            tensors, embed_module_path="model.decoder.embed_tokens"
        )
        self._add_lm_head_tensors(tensors)
        return tensors

    def _target_enabled(self, suffix: str) -> bool:
        return suffix in set(self._build_target_modules())

    def _add_embedding_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        embed_module_path: str,
    ) -> None:
        if self.vocab_size is None or not self._target_enabled("embed_tokens"):
            return
        lora_A = torch.zeros(self.rank, self.vocab_size, dtype=torch.float16)
        lora_B = torch.zeros(self.embedding_dim, self.rank, dtype=torch.float16)
        tensors[f"base_model.model.{embed_module_path}.lora_embedding_A"] = lora_A
        tensors[f"base_model.model.{embed_module_path}.lora_embedding_B"] = lora_B

    def _add_lm_head_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        if self.vocab_size is None or not self._target_enabled("lm_head"):
            return
        lora_A = torch.zeros(self.rank, self.embedding_dim, dtype=torch.float16)
        lora_B = torch.zeros(self.vocab_size, self.rank, dtype=torch.float16)
        tensors["base_model.model.lm_head.lora_A.weight"] = lora_A
        tensors["base_model.model.lm_head.lora_B.weight"] = lora_B

    def _build_config(self) -> dict:
        target_modules = self._build_target_modules()
        return {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": self.base_model_name,
            "bias": "none",
            "exclude_modules": [],
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": float(self.rank),
            "lora_dropout": 0.0,
            "megatron_core": "megatron.core",
            "megatron_config": None,
            "modules_to_save": None,
            "r": self.rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }

    def _set_ids_for_step(self, step: int) -> None:
        self.plus_id = self._base_plus_id
        self.minus_id = self._base_minus_id
        self.plus_name = "lozo_plus"
        self.minus_name = "lozo_minus"
        self.plus_path = self._request_path(self.plus_id)
        self.minus_path = self._request_path(self.minus_id)

    def _load_gpu_lora_pair(
        self,
        config: dict,
        plus_tensors: Dict[str, torch.Tensor],
        minus_tensors: Dict[str, torch.Tensor],
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("direct LoRA slots require LoRAUpdateRuntime(llm=...)")
        fn = partial(
            _load_two_lora_tensors_into_vllm_model,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            config=config,
            plus_tensors=plus_tensors,
            minus_tensors=minus_tensors,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError(
                "vLLM apply_model returned no results while loading LoRA"
            )
        return {"workers": results}

    def _update_gpu_lora_slots_direct(
        self,
        plus_A: Dict[str, torch.Tensor],
        plus_B: Dict[str, torch.Tensor],
        minus_A: Dict[str, torch.Tensor],
        minus_B: Dict[str, torch.Tensor],
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("direct LoRA slots require LoRAUpdateRuntime(llm=...)")
        fn = partial(
            _update_lora_slots_in_vllm_model,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            plus_A=plus_A,
            plus_B=plus_B,
            minus_A=minus_A,
            minus_B=minus_B,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError(
                "vLLM apply_model returned no results while updating LoRA"
            )
        return {"workers": results}

    def _update_gpu_lora_slots_from_directions(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        eps: float,
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("direct LoRA slots require LoRAUpdateRuntime(llm=...)")
        fn = partial(
            _update_lora_slots_from_directions_in_vllm_model,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            directions_2d=directions_2d,
            eps=eps,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError(
                "vLLM apply_model returned no results while updating LoRA"
            )
        return {"workers": results}

    def _update_gpu_lora_slot_from_direction(
        self,
        *,
        lora_id: int,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        eps: float,
        sign: float = 1.0,
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("direct LoRA slots require LoRAUpdateRuntime(llm=...)")
        fn = partial(
            _update_lora_slot_from_direction_in_vllm_model,
            lora_id=int(lora_id),
            directions_2d=directions_2d,
            eps=float(eps),
            sign=float(sign),
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError(
                "vLLM apply_model returned no results while updating one LoRA slot"
            )
        return {"workers": results}

    def _remove_gpu_lora(self, lora_id: int) -> None:
        if self.llm is None:
            return
        fn = partial(_remove_lora_from_vllm_model, lora_id=lora_id)
        self.llm.apply_model(fn)

    def register_slots(self):
        """
        Register two empty LoRA slots.

        Note: Actual tensors will be updated via update_plus_minus().
        """
        self._set_ids_for_step(step=0)
        config = self._build_config()
        empty_tensors = self._build_empty_tensors()

        self.last_update_info = self._load_gpu_lora_pair(
            config,
            empty_tensors,
            empty_tensors.copy(),
        )
        self._registered = True

    def build_peft_tensors(
        self,
        layer_to_A: Dict[str, torch.Tensor],
        layer_to_B: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert layer_to_A/B to PEFT format tensor dict.

        PEFT format: base_model.model.{module_path}.lora_A.weight

        Args:
            layer_to_A: {module_name: [rank, in_features]}  (HF weight names with .weight suffix)
            layer_to_B: {module_name: [out_features, rank]}

        Returns:
            tensors: {base_model.model.{module_path}.lora_A.weight: tensor, ...}
        """
        tensors = {}

        for name, A in layer_to_A.items():
            # Strip ".weight" suffix to get module path
            # e.g., "model.decoder.layers.0.self_attn.k_proj.weight" -> "model.decoder.layers.0.self_attn.k_proj"
            module_path = name
            if module_path.endswith(".weight"):
                module_path = module_path[:-7]  # Remove ".weight"

            if module_path.endswith(".embed_tokens"):
                tensors[f"base_model.model.{module_path}.lora_embedding_A"] = A
                tensors[f"base_model.model.{module_path}.lora_embedding_B"] = (
                    layer_to_B[name]
                )
            else:
                tensors[f"base_model.model.{module_path}.lora_A.weight"] = A
                tensors[f"base_model.model.{module_path}.lora_B.weight"] = layer_to_B[
                    name
                ]

        return tensors

    def update_plus_minus(
        self,
        plus_A: Dict[str, torch.Tensor],
        plus_B: Dict[str, torch.Tensor],
        minus_A: Dict[str, torch.Tensor],
        minus_B: Dict[str, torch.Tensor],
        step: int = 0,
    ):
        """
        Update both plus/minus LoRA slots.

        This overwrites the fixed plus/minus slots in place, so scoring requests
        only select already-loaded adapter IDs.
        """
        self._set_ids_for_step(step)
        if not self._registered:
            config = self._build_config()
            plus_tensors = self.build_peft_tensors(plus_A, plus_B)
            minus_tensors = self.build_peft_tensors(minus_A, minus_B)
            self.last_update_info = self._load_gpu_lora_pair(
                config,
                plus_tensors,
                minus_tensors,
            )
            self._registered = True
        else:
            self.last_update_info = self._update_gpu_lora_slots_direct(
                plus_A,
                plus_B,
                minus_A,
                minus_B,
            )

        self._registered = True

    def update_plus_minus_from_directions(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        *,
        eps: float,
        step: int = 0,
    ):
        """Update plus/minus GPU slots directly from U/V directions."""
        self._set_ids_for_step(step)
        if not self._registered:
            config = self._build_config()
            empty_tensors = self._build_empty_tensors()
            self.last_update_info = self._load_gpu_lora_pair(
                config,
                empty_tensors,
                empty_tensors.copy(),
            )
            self._registered = True
        self.last_update_info = self._update_gpu_lora_slots_from_directions(
            directions_2d,
            eps,
        )

    def update_slot_from_direction(
        self,
        *,
        lora_id: int,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        eps: float,
        sign: float = 1.0,
    ):
        """Update one registered direct LoRA slot from one direction map."""
        if not self._registered:
            self.register_slots()
        self.last_update_info = self._update_gpu_lora_slot_from_direction(
            lora_id=int(lora_id),
            directions_2d=directions_2d,
            eps=float(eps),
            sign=float(sign),
        )
        return self.last_update_info

    def get_plus_request_info(self) -> tuple[str, int, str]:
        """Get (name, id, path) for plus LoRA."""
        return self.plus_name, self.plus_id, self.plus_path

    def get_minus_request_info(self) -> tuple[str, int, str]:
        """Get (name, id, path) for minus LoRA."""
        return self.minus_name, self.minus_id, self.minus_path

    def cleanup(self):
        """Unregister LoRA slots."""
        if self._registered:
            self._remove_gpu_lora(self.plus_id)
            self._remove_gpu_lora(self.minus_id)
            self._registered = False
