main.pdf: main.tex $(wildcard sections/*.tex) refs.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf
