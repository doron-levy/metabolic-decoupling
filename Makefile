# Reproduce every data figure in the paper.
PY ?= python3

all: figures/shuttle_severing.pdf figures/shuttle_operator.pdf \
     figures/shuttle_growing.pdf figures/shuttle_fold.pdf

figures/shuttle_severing.pdf: code/sweeps.npz
	cd code && $(PY) plot.py
code/sweeps.npz: code/run_all.py code/shuttle.py
	cd code && $(PY) run_all.py

figures/shuttle_operator.pdf: code/Gmeasured.npz
	cd code && $(PY) fold_from_measured.py
code/Gmeasured.npz: code/full_solver.py
	cd code && $(PY) full_solver.py

figures/shuttle_growing.pdf: code/held_sweep.npz
	cd code && $(PY) plot_growing.py
code/held_sweep.npz: code/growing_run.py
	cd code && $(PY) growing_run.py

figures/shuttle_fold.pdf: code/plot_fold.py code/reduced_fold.py
	cd code && $(PY) plot_fold.py

clean:
	rm -f code/*.npz figures/*.pdf
.PHONY: all clean
