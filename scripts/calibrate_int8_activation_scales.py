"""Apply INT8 activation-scale metadata to an exported INT16 bundle.

This is intentionally a metadata calibration step, not a retraining pass. It
updates 1x1 Tensor Core candidate layers with conservative power-of-two
activation scales so the experimental INT8 route can be enabled explicitly via
runtime gates while the pure INT16 profile remains unchanged by default.
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply experimental INT8 activation scales to an INT16 bundle."
    )
    parser.add_argument("--input_bundle", required=True, help="Source .pt bundle.")
    parser.add_argument("--output_bundle", required=True, help="Destination .pt bundle.")
    parser.add_argument(
        "--stats_json",
        default=None,
        help="Optional JSON mapping layer names to max_abs values or scale values.",
    )
    parser.add_argument(
        "--stats_csv",
        default=None,
        help="Optional CSV with layer/module/path plus max_abs or activation_int8_scale columns.",
    )
    parser.add_argument(
        "--default_scale",
        type=int,
        default=1,
        help="Fallback activation scale for eligible layers missing stats.",
    )
    parser.add_argument(
        "--scale_key",
        default="activation_int8_scale",
        help="Stats field name to treat as a precomputed scale.",
    )
    parser.add_argument(
        "--max_abs_key",
        default="max_abs",
        help="Stats field name to convert into a power-of-two scale.",
    )
    return parser.parse_args()


def next_power_of_two(value):
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def scale_from_max_abs(max_abs):
    # Values are represented as int16 activations and cast down to int8 before
    # Tensor Core GEMM. The scale is the divisor that maps max_abs into +/-127.
    max_abs = abs(float(max_abs))
    if max_abs <= 127.0:
        return 1
    return next_power_of_two(int((max_abs + 126.0) // 127.0))


def normalize_name(name):
    return str(name or "").replace(".params", "").replace(".down_params", ".down")


def load_stats_json(path, scale_key, max_abs_key):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    rows = payload.values() if isinstance(payload, dict) and "rows" not in payload else payload.get("rows", [])
    if isinstance(payload, dict) and "rows" not in payload:
        return {
            normalize_name(name): _row_to_scale(row, scale_key, max_abs_key)
            for name, row in payload.items()
        }
    return _rows_to_scales(rows, scale_key, max_abs_key)


def load_stats_csv(path, scale_key, max_abs_key):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fp:
        return _rows_to_scales(csv.DictReader(fp), scale_key, max_abs_key)


def _row_to_scale(row, scale_key, max_abs_key):
    if isinstance(row, (int, float)):
        return scale_from_max_abs(row)
    if scale_key in row and row[scale_key] not in (None, ""):
        return max(1, int(float(row[scale_key])))
    if max_abs_key in row and row[max_abs_key] not in (None, ""):
        return scale_from_max_abs(row[max_abs_key])
    return None


def _rows_to_scales(rows, scale_key, max_abs_key):
    scales = {}
    for row in rows:
        name = row.get("layer") or row.get("module") or row.get("path") or row.get("name")
        scale = _row_to_scale(row, scale_key, max_abs_key)
        if name and scale is not None:
            scales[normalize_name(name)] = scale
    return scales


def iter_conv_params(node, path="root"):
    if isinstance(node, dict):
        if node.get("kind") == "conv2d" and "params" in node:
            yield path, node["params"]
        for key, value in node.items():
            yield from iter_conv_params(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for idx, value in enumerate(node):
            yield from iter_conv_params(value, f"{path}[{idx}]")


def is_int8_candidate(params):
    import torch

    weight = params.get("weight")
    return bool(
        isinstance(weight, torch.Tensor)
        and weight.dim() == 4
        and weight.shape[2:] == (1, 1)
        and params.get("stride", 1) == 1
        and params.get("padding", 0) == 0
        and params.get("groups", 1) == 1
        and weight.shape[0] % 4 == 0
        and weight.shape[1] % 4 == 0
        and params.get("weight_int8") is not None
    )


def apply_activation_scales(bundle, scales, default_scale):
    import torch

    updated = []
    models = bundle.get("models", bundle)
    for model_name, model_bundle in models.items():
        if not isinstance(model_bundle, dict):
            continue
        for path, params in iter_conv_params(model_bundle, model_name):
            if not is_int8_candidate(params):
                continue
            normalized = normalize_name(path)
            scale = int(scales.get(normalized, default_scale))
            scale = max(1, scale)
            params["activation_int8_scale"] = scale
            params["activation_int8_channel_scale"] = None
            params["eff_scale_c"] = None
            updated.append({"path": path, "activation_int8_scale": scale})
    bundle["int8_activation_calibration"] = {
        "updated_layers": updated,
        "default_scale": int(default_scale),
        "source": "metadata",
    }
    return bundle, updated


def main():
    args = parse_args()

    import torch

    scales = {}
    scales.update(load_stats_json(args.stats_json, args.scale_key, args.max_abs_key))
    scales.update(load_stats_csv(args.stats_csv, args.scale_key, args.max_abs_key))

    bundle = torch.load(args.input_bundle, map_location="cpu", weights_only=False)
    bundle, updated = apply_activation_scales(bundle, scales, args.default_scale)

    output_path = Path(args.output_bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(f"updated {len(updated)} INT8 activation scale entries")
    print(f"saved calibrated bundle to {output_path}")


if __name__ == "__main__":
    main()
