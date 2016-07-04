(TeX-add-style-hook
 "Diapos"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "francais") ("inputenc" "utf8") ("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "babel"
    "inputenc"
    "times"
    "fontenc")
   (LaTeX-add-bibitems
    "Author1990"
    "Someone2000"))
 :latex)

