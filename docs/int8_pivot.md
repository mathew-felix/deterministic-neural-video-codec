# INT8 Tensor Core Pivot

## Purpose

The INT8 path is an experimental acceleration route for eligible 1x1
convolutions. It is not the deterministic gold profile. Pure INT16 remains the
default path for bitstream-equivalence claims.

## Outcome

The INT8 route is retained as a documented engineering pivot, not as the
default runtime profile. The explored route can reduce arithmetic cost on
eligible 1x1 convolutions, but the closed-loop video codec is sensitive to
small reconstruction changes. Local errors can alter reference frames,
temporal prediction state, and entropy contexts in later P-frames.

The project therefore reinstates uncompromised pure INT16 as the gold profile.
INT8 experiments must be reported as separate runs with separate quality,
rate, and bitstream-equivalence evidence.

## Runtime Gates

INT8 Tensor Core kernels are disabled unless explicitly enabled:

- `DCVC_ENABLE_INT8_TC=1` enables eligible INT8 routes.
- `DCVC_INT8_SINGLE_LAYER=<module>` narrows routing to one layer for diagnosis.

The routing policy also blocks entropy-critical and reference-state-sensitive
modules. This is intentional: closed-loop video coding can amplify small local
numeric changes into future prediction and entropy-context drift.

Do not set either gate in reproducibility scripts for pure INT16 claims.

## Candidate Contract

A layer can use the INT8 prototype only when it is:

- a 1x1 `Conv2d`
- stride 1, padding 0, groups 1
- input and output channel counts divisible by 4
- exported with `weight_int8` metadata
- allowed by the runtime routing gate

The exporter packs INT16 weights into INT8 metadata and records per-layer
scaling information. `scripts/calibrate_int8_activation_scales.py` can then
apply conservative activation-scale metadata from observed maxima or
precomputed scale summaries.

Calibration does not make INT8 deterministic-equivalent by itself. It only
records metadata needed to run the experimental path in a controlled way.

## Why Pure INT16 Wins The Contract

The deterministic profile requires exact agreement in:

- quantized symbols
- scale-index lookup
- reconstructed reference frames
- DPB reset behavior
- entropy stream indexes
- final `.bin` bytes

INT8 Tensor Core execution changes intermediate arithmetic. Even if a frame
looks visually close, the changed reconstruction can feed the next P-frame and
invalidate the bitstream contract. For that reason, INT8 is useful for
performance research and failure analysis, while pure INT16 remains the
interoperability profile.

## Validation Rule

Any INT8 result must be reported separately from pure INT16. The minimum local
evidence is:

```powershell
python -m unittest discover -s tests -v
python tools/compare_bitstreams.py run_int16.bin run_int8.bin --expect_equal
```

The second command is expected to fail for many INT8 experiments. That failure
is useful evidence: it shows the fast path changed the bitstream contract.

For release or cross-device claims, run with `DCVC_ENABLE_INT8_TC` unset or set
to `0`.
