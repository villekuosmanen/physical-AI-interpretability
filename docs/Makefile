# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build HTML documentation
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Serve documentation locally
serve: html
	@echo "Starting local documentation server..."
	@echo "Documentation will be available at: http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	cd $(BUILDDIR)/html && python -m http.server 8000

# Build and serve in one command
docs-serve: serve

# Clean build artifacts
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Regenerate API docs and build
api-rebuild:
	sphinx-apidoc --force -a -o source .. ../physical_ai_interpretability
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@touch "$(BUILDDIR)/html/.nojekyll"

# Build for GitHub Pages deployment
github-pages: api-rebuild
	@echo "Documentation built for GitHub Pages deployment"
	@echo "Built files are in $(BUILDDIR)/html/"

.PHONY: help html serve docs-serve clean api-rebuild github-pages Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
