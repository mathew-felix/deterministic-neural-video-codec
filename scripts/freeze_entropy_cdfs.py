import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze DCVC entropy CDF tables into a reusable cross-device artifact."
    )
    parser.add_argument("--model_path_i", type=str, required=True)
    parser.add_argument("--model_path_p", type=str, required=False, default=None)
    parser.add_argument("--force_zero_thres", type=float, default=None, required=False)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def build_model(model_cls, ckpt_path, device, force_zero_thres):
    import torch

    from src.utils.common import get_state_dict

    model = model_cls()
    state_dict = get_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.update(force_zero_thres)
    return model


def main():
    args = parse_args()

    import torch

    from src.models.image_model import DMCI
    from src.models.video_model import DMC
    from src.utils.common import set_torch_env

    set_torch_env()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    payload = {
        "format_version": 1,
        "force_zero_thres": args.force_zero_thres,
        "models": {},
    }

    i_frame_model = build_model(DMCI, args.model_path_i, device, args.force_zero_thres)
    payload["models"]["i_frame_net"] = i_frame_model.export_frozen_entropy_state()

    if args.model_path_p is not None:
        p_frame_model = build_model(DMC, args.model_path_p, device, args.force_zero_thres)
        payload["models"]["p_frame_net"] = p_frame_model.export_frozen_entropy_state()

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"saved frozen entropy tables to {args.output_path}")


if __name__ == "__main__":
    main()
