.PHONY: setup build test test-cuda preflight encode decode pipeline manifest calibrate clean

setup:
	python -m venv venv
	venv\\Scripts\\python -m pip install -r requirements.txt

build:
	python bootstrap_runtime.py
	python build_int16_cuda.py

test:
	python -m pytest tests\\ -x -v --ignore=tests\\test_int16_kernels.py -k "not cuda"

test-cuda:
	python -m pytest tests\\ -x -v

preflight:
	python encode_mp4_to_bin.py --check_only

encode:
	python encode_mp4_to_bin.py

decode:
	python decode_bin_to_mp4.py --input_bin outputs\\smoke\\test_1280x720_30_32f_q32.bin

manifest:
	python scripts\\build_manifest.py

calibrate:
	python scripts\\calibrate_int16_bundle.py

pipeline:
	python pipeline.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['outputs','artifacts','nograph_out','__pycache__']]"
