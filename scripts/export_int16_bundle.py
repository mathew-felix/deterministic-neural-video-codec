import argparse
import os


GLOBAL_K2 = 8192
INT8_SAFE_ABS_MAX = 120


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DCVC models into an int16 reference bundle."
    )
    parser.add_argument("--model_path_i", type=str, required=True)
    parser.add_argument("--model_path_p", type=str, required=False, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--frozen_entropy_path", type=str, required=False, default=None)
    parser.add_argument("--force_zero_thres", type=float, default=None, required=False)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def get_frozen_entropy_state(blob, key, expected_type):
    if blob is None:
        return None
    if "models" in blob:
        return blob["models"].get(key)
    if blob["model_type"] != expected_type:
        raise RuntimeError(
            f"Frozen entropy state type mismatch for {key}: "
            f"expected {expected_type}, got {blob['model_type']}"
        )
    return blob


def build_model(model_cls, ckpt_path, device, force_zero_thres, frozen_state=None):
    import torch

    from src.utils.common import get_state_dict

    model = model_cls()
    state_dict = get_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.update(force_zero_thres=force_zero_thres, frozen_state=frozen_state)
    return model


def _iter_conv_params(node, path="root"):
    if isinstance(node, dict):
        if node.get("kind") == "conv2d" and "params" in node:
            yield path, node["params"]
        for key, value in node.items():
            yield from _iter_conv_params(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for idx, value in enumerate(node):
            yield from _iter_conv_params(value, f"{path}[{idx}]")


def compute_per_layer_k2(weight_int16, global_k2=GLOBAL_K2, int8_max=INT8_SAFE_ABS_MAX):
    import torch

    if not isinstance(weight_int16, torch.Tensor):
        raise TypeError("weight_int16 must be a torch.Tensor")
    if weight_int16.numel() == 0:
        return int(global_k2)

    max_abs = int(weight_int16.to(torch.int32).abs().max().item())
    if max_abs <= int8_max:
        return int(global_k2)

    k2_layer = int(int8_max * int(global_k2) / max_abs)
    return max(1, min(int(global_k2), k2_layer))


def _is_plan8_int8_candidate(params):
    import torch

    weight = params.get("weight")
    return bool(
        isinstance(weight, torch.Tensor)
        and weight.dim() == 4
        and weight.shape[2] == 1
        and weight.shape[3] == 1
        and params.get("groups", 1) == 1
        and params.get("stride", 1) == 1
        and params.get("padding", 0) == 0
        and weight.shape[0] % 4 == 0
        and weight.shape[1] % 4 == 0
    )


def _requantize_for_k2(weight_int16, bias_int32, k2_layer, global_k2=GLOBAL_K2):
    import torch

    if int(k2_layer) == int(global_k2):
        weight_rescaled = weight_int16.to(torch.int16)
        bias_rescaled = None if bias_int32 is None else bias_int32.to(torch.int32)
        return weight_rescaled, bias_rescaled

    ratio = float(k2_layer) / float(global_k2)
    weight_rescaled = torch.round(weight_int16.to(torch.float32) * ratio).clamp(
        -32767, 32767
    ).to(torch.int16)
    bias_rescaled = None
    if bias_int32 is not None:
        bias_rescaled = torch.round(bias_int32.to(torch.float32) * ratio).to(torch.int32)
    return weight_rescaled, bias_rescaled


def pack_weights_to_int8(model_bundle):
    import torch

    from src.layers.int16_backend import (
        iter_conv_param_specs,
        pack_weight_to_int8,
    )

    summary = []
    for path, params in iter_conv_param_specs(model_bundle, model_bundle.get("model_type", "model")):
        weight = params.get("weight")
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int16:
            continue
        tensor_core_eligible = _is_plan8_int8_candidate(params)
        k2_layer = int(params.get("k2_layer", GLOBAL_K2))
        if tensor_core_eligible:
            k2_layer = compute_per_layer_k2(weight, global_k2=GLOBAL_K2)
            weight_rescaled, bias_rescaled = _requantize_for_k2(
                weight,
                params.get("bias"),
                k2_layer,
                global_k2=GLOBAL_K2,
            )
            weight_i8 = weight_rescaled.clamp(-127, 127).to(torch.int8)
            params["weight"] = weight_rescaled
            params["bias"] = bias_rescaled
            params["k2_layer"] = k2_layer
            params["weight_int8"] = weight_i8
            params["weight_int8_scale"] = 1
            params["weight_int8_channel_scale"] = None
            params["activation_int8_scale"] = 1
            params["activation_int8_channel_scale"] = None
            params["eff_scale_c"] = None
            scale_mean = 1.0
            scale_max = 1
            pct_eq_1 = 100.0
            packed_min = int(weight_rescaled.min().item())
            packed_max = int(weight_rescaled.max().item())
        else:
            weight_i8, secondary_scale = pack_weight_to_int8(weight)
            params["weight_int8"] = weight_i8
            params["weight_int8_scale"] = secondary_scale
            params["weight_int8_channel_scale"] = None
            params["activation_int8_channel_scale"] = None
            params["eff_scale_c"] = None
            params["activation_int8_scale"] = 1
            scale_mean = float(secondary_scale)
            scale_max = int(secondary_scale)
            pct_eq_1 = 100.0 if secondary_scale == 1 else 0.0
            packed_min = int(weight.min().item())
            packed_max = int(weight.max().item())
        summary.append(
            {
                "path": path,
                "shape": list(weight.shape),
                "min": packed_min,
                "max": packed_max,
                "k2_layer": k2_layer,
                "secondary_scale_mean": scale_mean,
                "secondary_scale_max": scale_max,
                "pct_eq_1": pct_eq_1,
                "tensor_core_eligible": tensor_core_eligible,
            }
        )
    return summary


def main():
    args = parse_args()

    import torch

    from src.models.image_model import DMCI
    from src.models.int16_reference import export_dmci_int16_bundle, export_dmc_int16_bundle
    from src.models.video_model import DMC
    from src.utils.common import set_torch_env

    set_torch_env()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    frozen_blob = None
    if args.frozen_entropy_path is not None:
        frozen_blob = torch.load(
            args.frozen_entropy_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    payload = {
        "format_version": 3,
        "force_zero_thres": args.force_zero_thres,
        "source": {
            "model_path_i": args.model_path_i,
            "model_path_p": args.model_path_p,
            "frozen_entropy_path": args.frozen_entropy_path,
        },
        "models": {},
    }

    i_frame_model = build_model(
        DMCI,
        args.model_path_i,
        device,
        args.force_zero_thres,
        frozen_state=get_frozen_entropy_state(frozen_blob, "i_frame_net", "DMCI"),
    )
    payload["models"]["i_frame_net"] = export_dmci_int16_bundle(i_frame_model)
    i_frame_summary = pack_weights_to_int8(payload["models"]["i_frame_net"])
    payload["dmci_int8"] = payload["models"]["i_frame_net"]
    payload["dmci_weight_packing"] = i_frame_summary

    if args.model_path_p is not None:
        p_frame_model = build_model(
            DMC,
            args.model_path_p,
            device,
            args.force_zero_thres,
            frozen_state=get_frozen_entropy_state(frozen_blob, "p_frame_net", "DMC"),
        )
        payload["models"]["p_frame_net"] = export_dmc_int16_bundle(p_frame_model)
        p_frame_summary = pack_weights_to_int8(payload["models"]["p_frame_net"])
        payload["dmc_int8"] = payload["models"]["p_frame_net"]
        payload["dmc_weight_packing"] = p_frame_summary
    else:
        p_frame_summary = []

    print("INT8 packing summary:")
    for entry in i_frame_summary + p_frame_summary:
        print(
            f"{entry['path']}: range [{entry['min']}, {entry['max']}] "
            f"k2={entry['k2_layer']} "
            f"S_W_c_mean={entry['secondary_scale_mean']:.1f} "
            f"S_W_c_max={entry['secondary_scale_max']} "
            f"pct_eq_1={entry['pct_eq_1']:.0f}% "
            f"tc_eligible={entry['tensor_core_eligible']}"
        )

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"saved int16 reference bundle to {args.output_path}")


if __name__ == "__main__":
    main()
