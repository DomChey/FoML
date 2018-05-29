(TeX-add-style-hook
 "template_Report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("natbib" "square" "sort" "comma" "numbers")))
   (TeX-run-style-hooks
    "latex2e"
    "report"
    "rep11"
    "inputenc"
    "algorithm"
    "algpseudocode"
    "hyperref"
    "graphicx"
    "subcaption"
    "booktabs"
    "natbib")
   (LaTeX-add-labels
    "eq:dissimilarity"
    "eq:compatibility"
    "eq:mutualComp"
    "algo:placer"
    "img:432"
    "img:540"
    "img:805"
    "img:2360"
    "fig:database"
    "Results"
    "img:540_8"
    "img:540_8_reconstructed"
    "fig:fail")
   (LaTeX-add-bibliographies
    "literature"))
 :latex)

