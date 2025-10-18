# E-PSS

**A project in which we extend the Polarimetric Slope Sensing (PSS) technique for remotely observing ocean waves.**

Manuscript in preparation for *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (*J-STARS*):

"The Extended Polarimetric Slope Sensing Technique"
Nathan J. M. Laxague, Z. Göksu Duvarci, Lindsay Hogan, Junzhe Liu, Christopher Bouillon, and Christopher J. Zappa

## First Steps

Clone the repository:
```
git clone https://github.com/unh-cassll/E-PSS_paper.git
cd E-PSS_paper
```

Create a Python virtual environment and install the dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Grab the observational data from its public repository
```
**todo**
```

Make the figures:
```
make figures
```

Make the paper:
```
make paper
```