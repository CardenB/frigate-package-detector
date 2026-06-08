.PHONY: help download convert build data analyze balance baseline benchmark report train resume export clean clean-data \
        frigate-pull frigate-review frigate-export wandb-up wandb-down
PY := python scripts
ARGS ?=

help:
	@echo "Targets:"
	@echo "  make download   - fetch all enabled sources into data/raw/"
	@echo "  make convert    - normalize raw -> data/interim/ (unified labels)"
	@echo "  make build      - merge + split -> data/final/ (+ data.yaml)"
	@echo "  make data       - download + convert + build (full pipeline)"
	@echo "  make analyze    - per-class box counts / balance report"
	@echo "  make balance    - TRAIN class balance before vs after oversampling"
	@echo "  make baseline   - eval stock yolov9s on COCO val (forgetting reference)"
	@echo "  make benchmark  - model x scenario matrix (ARGS='--scenario package') -> report"
	@echo "  make report     - regenerate side-by-side benchmark report (md + html)"
	@echo "  make train      - fine-tune yolov9s (auto-resume on crash; auto-starts wandb)"
	@echo "  make resume     - resume the named run from last.pt (after a reboot/kill)"
	@echo "  make export     - export best.pt -> ONNX + labelmap for Frigate"
	@echo "  make wandb-up    - ensure self-hosted wandb server is running (:8080)"
	@echo "  make wandb-down  - stop the wandb server container"
	@echo "  --- review flywheel (your own camera data) ---"
	@echo "  make frigate-pull    - pull Frigate detections into a review round (ARGS=...)"
	@echo "  make frigate-review  - review/tag a round in FiftyOne (ARGS='--round <id>')"
	@echo "  make frigate-export  - reviewed round -> data/raw/frigate/ (then make convert build)"
	@echo "  make clean-data - remove data/{raw,interim,final} contents"

download:
	$(PY)/download_package_seg.py
	$(PY)/download_open_images.py
	$(PY)/download_coco_subset.py
	$(PY)/download_roboflow.py

convert:
	$(PY)/convert_to_yolo.py

build:
	$(PY)/build_dataset.py

data: download convert build

analyze:
	$(PY)/analyze_dataset.py

balance:
	$(PY)/analyze_balance.py

baseline:
	$(PY)/eval_baseline.py
	$(PY)/make_benchmark_report.py

benchmark:
	$(PY)/run_benchmarks.py $(ARGS)

report:
	$(PY)/make_benchmark_report.py

train:
	$(PY)/train.py $(ARGS)

resume:
	$(PY)/train.py --resume $(ARGS)

export:
	$(PY)/export_onnx.py

wandb-up:
	$(PY)/ensure_wandb.py

wandb-down:
	docker stop wandb-local

frigate-pull:
	$(PY)/frigate_pull.py $(ARGS)

frigate-review:
	$(PY)/frigate_review.py $(ARGS)

frigate-export:
	$(PY)/frigate_export.py $(ARGS)

clean-data:
	rm -rf data/raw/* data/interim/* data/final/*
	@touch data/raw/.gitkeep data/interim/.gitkeep data/final/.gitkeep
