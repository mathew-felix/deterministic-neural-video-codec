# Contributing

Thank you for your interest in contributing to this project. This document
covers the workflow, code style, and determinism constraints that all changes
must satisfy.

---

## Determinism Contract

Every change that touches `src/` must preserve the **bitstream equivalence
contract**: given the same source video, model bundle, runtime flags, and codec
revision, the `.bin` output must be byte-identical across all supported NVIDIA
GPU architectures (Turing, Ampere, Ada).

Before opening a pull request, verify the contract by running the same encode
on two different GPUs or two different GPU driver versions and comparing the
bitstreams:

```bash
python tools/compare_bitstreams.py outputs/run_a/*.bin outputs/run_b/*.bin --expect_equal
```

Any change that breaks bitstream equivalence is a regression, regardless of
other improvements it may bring.

---

## Development Setup

```bash
git clone https://github.com/<your-username>/deterministic-neural-video-codec.git
cd deterministic-neural-video-codec

python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.txt
python bootstrap_runtime.py
python build_int16_cuda.py
```

Run the test suite before making any changes to confirm a clean baseline:

```bash
python -m pytest tests/ -x -v --ignore=tests/test_int16_kernels.py -k "not cuda"
```

---

## Workflow

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b fix/short-description
   ```

2. Make your changes. Keep commits small and scoped to a single concern.

3. Run linting and tests:
   ```bash
   python -m py_compile src/layers/int16_backend.py src/models/int16_reference.py
   python -m pytest tests/ -x -v --ignore=tests/test_int16_kernels.py -k "not cuda"
   ```

4. Update `CHANGELOG.md` with a brief entry under the `[Unreleased]` section.

5. Open a pull request against `main`. Describe what changed and why.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Maximum line length: 100 characters.
- All public functions and classes require a docstring.
- Comments explain *why*, not *what* — avoid restating what the code already
  says.
- No `TODO`, `FIXME`, or `HACK` comments without a linked issue number.

---

## What to Contribute

Pull requests are welcome for:

- Bug fixes with a reproducing test case
- Performance improvements that preserve the determinism contract
- Documentation improvements and correction of errors
- Additional test coverage for edge cases

Please open an issue before starting work on large features or architectural
changes, so we can discuss scope and approach first.

---

## Reporting Issues

Open a GitHub issue with:

1. The exact command you ran
2. The full error output
3. Your OS, GPU model, CUDA version, and PyTorch version
4. The output of `python encode_mp4_to_bin.py --check_only` (if applicable)
