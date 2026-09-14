# paper/

CVPR-format LaTeX skeleton for the EIP paper.

## Style files

`cvpr.sty` and `ieeenat_fullname.bst` are the real files fetched from the
official CVPR author kit (https://github.com/cvpr-org/author-kit), commit
current as of 2026-09-14. No fallback was needed.

Note: the bibliography style file in that kit is named `ieeenat_fullname.bst`
(not `ieee_fullname.bst`); `main.tex` and `Makefile` use the real name.

## Build status: NOT VERIFIED

`pdflatex` and `bibtex` are not installed on this machine (`which pdflatex
bibtex` found neither). `make -C paper` has not been run and `main.pdf` has
not been produced. Whoever runs the build first should treat that as the
first real verification of this skeleton, and fix any compile errors that
surface then (untested placeholder content is a plausible source of them).

## Section ownership

`sections/*.tex` are placeholders. `sections/method.tex` is deliberately
empty — the Method section is out of scope until the method is finalized.
Every other section file carries a `% TODO: Task N` comment naming the task
that fills it in (see the plan at
`.superpowers/sdd/2026-09-14-eip-paper-writing/`).
