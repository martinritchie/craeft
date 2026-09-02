// MathJax configuration for the docs site.
//
// Arithmatex (generic mode) fences inline math as \( \) and display math
// as \[ \]; MathJax renders it. `tags: "ams"` numbers only equations
// written inside \begin{equation} ... \end{equation}, and \label / \eqref
// then work as in LaTeX. Numbering restarts on every page.
window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};
