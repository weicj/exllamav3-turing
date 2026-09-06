from __future__ import annotations
from typing_extensions import override
import hashlib
import json
import os
import torch
from ..util.tensor import to2
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP, Linear
from .hyperconnections import HyperConnection
from ..util import profile_opt


_subtrace_dir = os.environ.get("EXL3_SUBTRACE_DIR")
_subtrace_path = os.environ.get("EXL3_SUBTRACE_JSONL")
_subtrace_layer = int(os.environ.get("EXL3_SUBTRACE_LAYER", "-1"))
_subtrace_layers = {
    int(value) for value in os.environ.get("EXL3_SUBTRACE_LAYERS", "").split(",") if value
}
_subtrace_stages = {
    value for value in os.environ.get("EXL3_SUBTRACE_STAGES", "").split(",") if value
}
_subtrace_signature = os.environ.get("EXL3_SUBTRACE_SIGNATURE", "0") == "1"
_subtrace_call_filter = {
    int(value) for value in os.environ.get("EXL3_SUBTRACE_CALLS", "").split(",") if value
}
_subtrace_calls = {}
_subtrace_current_calls = {}


def _subtrace_sha256(tensor: torch.Tensor) -> str:
    # A full cryptographic digest requires copying every traced tensor to the host. For
    # a broad layer scan, use exact integer summaries of the raw bytes on the GPU instead.
    # Four residue classes plus byte squares make an accidental collision implausible for
    # numerical-drift localization, while keeping PCIe traffic to a few scalar values.
    if _subtrace_signature:
        bits = tensor.detach().contiguous().view(torch.uint8)
        sums = []
        squares = []
        for offset in range(4):
            values = bits[offset::4].to(torch.int64)
            sums.append(int(values.sum().item()))
            squares.append(int(values.square().sum().item()))
        return "sig:" + ",".join(str(value) for value in (*sums, *squares))
    bits = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(bits).hexdigest()


def _save_subtrace(module, stage: str, tensor: torch.Tensor):
    """Save selected TransformerBlock boundaries during a Qwen4Exp TP parity run."""
    if (_subtrace_dir is None and _subtrace_path is None) or (
        module.layer_idx not in _subtrace_layers if _subtrace_layers else module.layer_idx != _subtrace_layer
    ):
        return
    # Mark the invocation before any sublayer mixer consumes the residual stack.
    # Updating before stage selection lets a trace select a prefill/decode call even
    # when it records only later boundaries.
    if stage == "block_input":
        call = _subtrace_calls.get(module.key, 0)
        _subtrace_calls[module.key] = call + 1
        _subtrace_current_calls[module.key] = call
    else:
        call = _subtrace_current_calls.get(module.key, -1)
    if _subtrace_call_filter and call not in _subtrace_call_filter:
        return
    if _subtrace_stages and stage not in _subtrace_stages:
        return
    if _subtrace_path is not None:
        row = {
            "call": call,
            "key": module.key,
            "stage": stage,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _subtrace_sha256(tensor),
        }
        # TP workers inherit one environment. Keep each rank's hash log separate so
        # concurrent appends cannot interleave JSON records.
        trace_path = f"{_subtrace_path}.d{tensor.device.index}"
        with open(trace_path, "a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if _subtrace_dir is not None:
        os.makedirs(_subtrace_dir, exist_ok = True)
        mode = os.environ.get("EXL3_TRACE_MODE", "unknown")
        device = str(tensor.device).replace(":", "_")
        torch.save({
            "key": module.key,
            "call": call,
            "stage": stage,
            "shape": tuple(tensor.shape),
            "tensor": tensor.detach().float().cpu().contiguous(),
        }, os.path.join(
            _subtrace_dir,
            f"{mode}-{device}-l{module.layer_idx:03d}-c{call:03d}-{stage}.pt",
        ))

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        attn_hc: HyperConnection | None = None,
        mlp_hc: HyperConnection | None = None,
        key_layer_scalar: str | None = None,
        key_attn_resid_scalar: str | None = None,
        key_mlp_resid_scalar: str | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.attn_hc = attn_hc
        self.mlp_hc = mlp_hc
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.key_layer_scalar = key_layer_scalar
        self.key_attn_resid_scalar = key_attn_resid_scalar
        self.key_mlp_resid_scalar = key_mlp_resid_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

        # Hyperconnection sites (mHC): the block's residual is (bsz, seq, hc_mult, hidden)
        # fp32 streams, mixed at each sublayer site instead of the plain residual add
        if attn_hc is not None or mlp_hc is not None:
            assert attn_hc is not None and mlp_hc is not None, \
                "hyperconnections require both attn_hc and mlp_hc"
            assert all(v is None for v in (
                attn_post_norm, mlp_post_norm,
                key_layer_scalar, key_attn_resid_scalar, key_mlp_resid_scalar,
            )), \
                "hyperconnections cannot combine with residual scalars/post-norms"

        self.register_submodule(self.attn_hc)
        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_hc)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()

        # TODO: Residual scalar tensors could be baked into preceding modules for models that use them
        #       (currently only Step3.7 vision tower)
        if self.key_attn_resid_scalar:
            self.attn_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_attn_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )
        if self.key_mlp_resid_scalar:
            self.mlp_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_mlp_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )

    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

    def get_tensors(self):
        t = {}
        if self.key_layer_scalar is not None:
            t[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        if self.key_attn_resid_scalar is not None:
            t[self.key + "." + self.key_attn_resid_scalar] = self.attn_resid_scalar.data.contiguous()
        if self.key_mlp_resid_scalar is not None:
            t[self.key + "." + self.key_mlp_resid_scalar] = self.mlp_resid_scalar.data.contiguous()
        return t

    def weights_numel(self):
        return (
            super().weights_numel() +
            (1 if self.key_layer_scalar is not None else 0) +
            (self.attn_resid_scalar.numel() if self.attn_resid_scalar is not None else 0) +
            (self.mlp_resid_scalar.numel() if self.mlp_resid_scalar is not None else 0)
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        # An opt-in TP worker profile supplies an event list for one requested forward pass.
        # Keep the instrumentation here, at the architectural boundaries, rather than inside
        # individual kernels: this gives a reliable GDN/attention vs. MoE vs. hyperconnection
        # breakdown while retaining the normal asynchronous execution schedule.
        profile_events = params.get("_tp_submodule_profile")

        def profile_start():
            if profile_events is None:
                return None
            event = torch.cuda.Event(enable_timing = True)
            event.record()
            return event

        def profile_end(stage: str, start):
            if start is None:
                return
            end = torch.cuda.Event(enable_timing = True)
            end.record()
            profile_events.append((self.key, stage, start, end))

        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0

        y_resid = None  # pending attn output whose residual add is folded into the MLP input norm

        if self.attn:
            _save_subtrace(self, "block_input", x)
            event = profile_start()
            if self.attn_hc:
                hc_post, hc_comb, y = self.attn_hc.mix(x, params)
                _save_subtrace(self, "attn_hc_mixed", y)
                y = y.half()
                if self.attn_norm:
                    y = self.attn_norm.forward(y, params, out_dtype = torch.half)
            elif self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            profile_end("attn_input", event)
            _save_subtrace(self, "attn_input", y)
            event = profile_start()
            y = self.attn.forward(y, params)
            profile_end("attn", event)
            _save_subtrace(self, "attn_output", y)
            if params.get("prefill") and not export_state:
                return x
            if self.attn_resid_scalar is not None:
                y *= self.attn_resid_scalar
            event = profile_start()
            if self.attn_hc:
                x = self.attn_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.attn_post_norm:
                self.attn_post_norm.forward(y, params, residual = x)
            elif self.mlp is not None and self.mlp_norm is not None and self.mlp_norm.can_fuse_residual(x, y):
                y_resid = y
            else:
                x += y
            profile_end("attn_residual", event)
            _save_subtrace(self, "post_attn", x)

        if self.mlp:
            event = profile_start()
            if self.mlp_hc:
                hc_post, hc_comb, y = self.mlp_hc.mix(x, params)
                _save_subtrace(self, "mlp_hc_mixed", y)
                y = y.half()
                if self.mlp_norm:
                    y = self.mlp_norm.forward(y, params, out_dtype = torch.half)
            else:
                params["residual"] = x
                if y_resid is not None:
                    y = self.mlp_norm.forward(y_resid, params, out_dtype = torch.half, residual_in = x)
                elif self.mlp_norm:
                    y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
                else:
                    y = x.half()
            profile_end("mlp_input", event)
            _save_subtrace(self, "mlp_input", y)
            event = profile_start()
            y = self.mlp.forward(y, params)
            profile_end("mlp", event)
            _save_subtrace(self, "mlp_output", y)
            if self.mlp_resid_scalar is not None:
                y *= self.mlp_resid_scalar
            event = profile_start()
            if self.mlp_hc:
                x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.mlp_post_norm:
                self.mlp_post_norm.forward(y, params, residual = x)
            else:
                x += y
            profile_end("mlp_residual", event)
            _save_subtrace(self, "post_mlp", x)

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            # With hyperconnections the residual is a stream stack; export the stream mean as
            # the collapsed hidden state (streams start as broadcast copies of the embedding)
            x_ = x.mean(dim = 2) if self.attn_hc else x
            if x_.dtype == torch.half:
                s.append(x_.clamp_(-65504.0, 65504.0))
            else:
                x_ = x_.half()
                x_.clamp_(-65504.0, 65504.0)
                s.append(x_)

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": TransformerBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
                "key_layer_scalar": self.key_layer_scalar,
                "key_attn_resid_scalar": self.key_attn_resid_scalar,
                "key_mlp_resid_scalar": self.key_mlp_resid_scalar,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_hc",
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_hc",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            # Per-layer scalars load from the tensor collection, which TP children don't have
            "layer_scalar_f": self.layer_scalar_f,
            "attn_resid_scalar": producer.send(self.attn_resid_scalar),
            "mlp_resid_scalar": producer.send(self.mlp_resid_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_hc = _import("attn_hc"),
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_hc = _import("mlp_hc"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )

        module.layer_scalar_f = exported.get("layer_scalar_f")
        module.attn_resid_scalar = consumer.recv(exported.get("attn_resid_scalar"), cuda = True)
        module.mlp_resid_scalar = consumer.recv(exported.get("mlp_resid_scalar"), cuda = True)
        module.device = device
        return module


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        input_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | None = None,
        mlp: MLP | GatedMLP | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.input_norm = input_norm
        self.attn = attn
        self.mlp = mlp
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.input_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        if not params.get("prefill"):
            y2 = self.mlp.forward(y, params)
            y1 += y2

            if self.tp_reduce:
                params["backend"].all_reduce(y1)

            x += y1

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": ParallelDecoderBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "input_norm",
                "attn",
                "mlp",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name, **kwargs):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, **kwargs) \
                if exported.get(name) else None

        module = ParallelDecoderBlock(
            config = None,
            **exported["kwargs"],
            input_norm = _import("input_norm"),
            attn = _import("attn", skip_reduction = True),
            mlp = _import("mlp", skip_reduction = True),
        )
        module.device = device

        # Use single reduction for sum of mlp and attn
        module.tp_reduce = True
        return module
