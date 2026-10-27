PY ?= python
.PHONY: all verify clean
all:
	PY=$(PY) sh run_all.sh
verify:
	$(PY) scripts/13_verify.py
clean:
	rm -rf work outputs logs
