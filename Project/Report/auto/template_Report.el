(TeX-add-style-hook
 "template_Report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8")))
   (TeX-run-style-hooks
    "latex2e"
    "report"
    "rep10"
    "inputenc")
   (LaTeX-add-bibliographies
    "literature"))
 :latex)

