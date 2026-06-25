\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{standalone}[2020/01/01 minimal standalone stub]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}
\RequirePackage[margin=0.5cm]{geometry}
\pagestyle{empty}
\setlength{\parindent}{0pt}
