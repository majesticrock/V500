all: build/main.pdf

build/plots.check: 5plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python 5plot.py

# hier Python-Skripte:
build/messung2.pdf: messung2.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python messung2.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf:	build/plots.check build/messung2.pdf content/tabelle.tex

build/main.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
