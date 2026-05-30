# E-PSS

**A project in which we extend the Polarimetric Slope Sensing (PSS) technique for remotely observing ocean waves.**

Manuscript submitted to *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*:

"E-PSS: the Extended Polarimetric Slope Sensing technique for measuring ocean surface waves"
Nathan J. M. Laxague, Z. Göksu Duvarci, Christopher Bouillon, Lindsay Hogan, Junzhe Liu, and Christopher J. Zappa

LaTeX source and Python scripts are licensed under [CC-BY](LICENSE).

## First Steps

Install the uv Python package manager
```
# see https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository:
```
git clone https://github.com/unh-cassll/E-PSS_paper.git
cd E-PSS_paper
```

Install Python, create a virtual environment, and install the dependencies:
```
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
make install
```

`make install` adds `--refresh-package` flags so each install brings in
GitHub HEAD of the two upstream packages we depend on (`ewdm` from
`dspelaez/extended-wdm`, `epss` from `unh-cassll/polarimetric-slope-sensing`).
**Do not run `uv pip install --force-reinstall epss` on its own** — that
ignores `requirements.txt`, so uv resolves `ewdm` from PyPI (currently 0.4,
which is missing the `np.trapz -> np.trapezoid` patch we rely on). To pull
the latest of just those two upstream packages without re-resolving the
rest of the environment, use `make update-deps`.

Grab data from public repository:
```
python3 _codes/grab_observational_data.py
```

Make the data (produce intermediate products):
```
make data
```

Make the figures:
```
make figures
```

Make the paper:
```
make paper
```

## Acknowledgments
The structure of this self-contained repository was inspired by [wave-modulation-paper](https://github.com/wavesgroup/wave-modulation-paper) by Prof. Milan Curcic at UMiami.
