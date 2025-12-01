SPHINXOPTS    =
SPHINXBUILD   = uv run sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  clean      to remove build files"

html:
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html."

clean:
	rm -rf $(BUILDDIR)/*

.PHONY: help html clean
