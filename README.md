# E-PSS

**A project in which we extend the Polarimetric Slope Sensing (PSS) technique for remotely observing ocean waves.**

Manuscript in preparation for *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (*J-STARS*):

"E-PSS: the Extended Polarimetric Slope Sensing technique for measuring ocean surface waves"
Nathan J. M. Laxague, Z. Göksu Duvarci, Lindsay Hogan, Junzhe Liu, Christopher Bouillon, and Christopher J. Zappa

LaTeX source and Python scripts are licensed under [CC-BY](LICENSE).

## First Steps

Install the uv Python package manager
```
git clone https://github.com/astral-sh/uv
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
uv pip install -r requirements.txt
```

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
