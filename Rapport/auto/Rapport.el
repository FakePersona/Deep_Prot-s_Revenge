(TeX-add-style-hook
 "Rapport"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "11pt" "onecolumn")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("inputenc" "utf8") ("babel" "main=francais" "english") ("biblatex" "backend=biber" "style=authoryear-comp")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "a4wide"
    "tabularx"
    "fullpage"
    "fontenc"
    "inputenc"
    "babel"
    "graphicx"
    "xspace"
    "float"
    "wrapfig"
    "url"
    "mathpazo"
    "biblatex"
    "listings"
    "svg")
   (TeX-add-symbols
    "manual"
    "dbend"
    "bfseriesaux")
   (LaTeX-add-environments
    "keywords"
    "remarque"
    "attention")
   (LaTeX-add-bibliographies
    "Biblio"))
 :latex)

