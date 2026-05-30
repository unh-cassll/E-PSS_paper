.PHONY: all clean figures data install update-deps

all: paper

# Install / refresh dependencies. --refresh-package forces uv to re-clone
# the git source for ewdm and epss even when they are already cached, so
# every `make install` brings in github HEAD. Use this instead of bare
# `uv pip install -r requirements.txt` to guarantee freshness, and NEVER
# run `uv pip install --force-reinstall epss` on its own -- that does NOT
# read requirements.txt, so uv resolves ewdm from PyPI (currently 0.4,
# which lacks the np.trapz -> np.trapezoid patch we depend on).
install:
	uv pip install \
	  --refresh-package ewdm \
	  --refresh-package epss \
	  -r requirements.txt

# Force a fresh pull of just the two github-hosted packages (use after a
# known upstream change without re-resolving the whole environment).
update-deps:
	uv pip install --upgrade \
	  --refresh-package ewdm --reinstall-package ewdm \
	  --refresh-package epss --reinstall-package epss \
	  "ewdm @ git+https://github.com/dspelaez/extended-wdm.git" \
	  "epss @ git+https://github.com/unh-cassll/polarimetric-slope-sensing.git#subdirectory=Python"

data:
	$(MAKE) -j 11 -C _data

figures:
	$(MAKE) -j 11 -C _codes

paper: paper.tex references.bib
	pdflatex -halt-on-error $@
	bibtex $@
	pdflatex -halt-on-error $@
	pdflatex -halt-on-error $@

clean:
	$(RM) *.aux *.bbl *.blg *.cut *fdb_latexmk *.fls *.log *.out *.pdf _figures/*.pdf
