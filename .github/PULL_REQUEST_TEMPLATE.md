## Summary
- Briefly describe what changed and why.

## Validation
- [ ] `python -m py_compile src/layers/int16_backend.py src/models/int16_reference.py encode_mp4_to_bin.py decode_bin_to_mp4.py`
- [ ] `python -m pytest tests/ -x -v --ignore=tests/test_int16_kernels.py -k "not cuda"`
- [ ] Determinism validated (if codec/runtime logic changed)

## Notes
Include any caveats or follow-up actions.
