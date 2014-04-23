# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
SRC_DIR       = source
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(SRC_DIR)

.PHONY: help clean html changes linkcheck api

default: html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html        standalone HTML files"
	@echo "  html_noapi  same as above, without the time consuming API docs"
	@echo "  changes     an overview over all changed/added/deprecated items"
	@echo "  linkcheck   check all external links for integrity (takes a long time)"
	@echo
	@echo "Compound utility targets:"
	@echo "pdf          latex and then runs the PDF generation"
	@echo "all          html and pdf"
	@echo "dist         all, and then puts the results in dist/"

clean_api:
	-rm -rf $(SRCDIR)/api/generated

clean: clean_api
	-rm -rf $(BUILDDIR)/*

pdf: latex
	cd $(BUILDDIR)/latex && make all-pdf

all: html pdf

html: api
html_noapi: clean_api

html html_noapi:
	mkdir -p $(BUILDDIR)/html $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

api: $(SRC_DIR)/api/generated/gen.txt

$(SRC_DIR)/api/generated/gen.txt:
	python autogen_api.py
	@echo "Build API docs finished."

latex: api
	mkdir -p $(BUILDDIR)/latex $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	mkdir -p $(BUILDDIR)/changes $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	mkdir -p $(BUILDDIR)/linkcheck $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.rst."
