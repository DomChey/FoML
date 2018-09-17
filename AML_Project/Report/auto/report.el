(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("natbib" "square" "sort" "comma" "numbers")))
   (TeX-run-style-hooks
    "latex2e"
    "rep11"
    "inputenc"
    "algorithm"
    "algpseudocode"
    "hyperref"
    "graphicx"
    "subcaption"
    "booktabs"
    "natbib")
   (LaTeX-add-bibliographies
    "literature"))
 :latex)

