IEEE Access manuscript source - availability update, 2026-10-31

Main entry: main.tex
Body: paper_body.tex
Compiler: pdfLaTeX; bibliography: BibTeX / IEEEtran.

Build from this directory with a normal TeX Live / MiKTeX installation:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The included latexmkrc adds template/ to the TeX, bibliography-style,
font-metric, Type 1 font, and font-map search paths. Do not disable this
configuration with -norc when building without equivalent environment paths.
On Overleaf, choose main.tex as the main document and pdfLaTeX as compiler.
All original template, font, figure, table, bibliography and author-image
files are retained. main.bbl is a generated convenience copy of the unchanged
bibliography; references.bib and template/IEEEtran.bst remain authoritative.
Standard LaTeX packages, including needspace, and the standard Courier font
package must be installed in the TeX distribution.

Changes relative to the originally supplied source ZIP (2026-10-31):
- The obsolete public-repository availability sentence in Reproducibility
  Details is updated; software versions and revision IDs are retained.
- Data and Code Availability is added immediately before Acknowledgment,
  using the exact author-provided English statement and a clickable URL.
- biographies.tex: remove the manual page break before the third biography
  so the remaining authors flow into the available final-column space.
  Biography wording, images, font sizes and author order are unchanged.
No title, author, affiliation, correspondence, experiment, result, method,
conclusion, table, figure, reference, or AI-assistance disclosure is changed.
