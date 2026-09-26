#!/usr/bin/env python3
"""Render the accompanying Markdown plan using local LaTeX; no third-party Python dependencies."""
from pathlib import Path
import re, subprocess

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'MadVoro_Performance_Plan.md'
OUT=ROOT/'MadVoro_Performance_Plan.tex'

def esc(s):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    s=s.replace('—',' -- ').replace('–','-').replace('’',"'").replace('“','``').replace('”',"''").replace('×',r' x ')
    out=''
    for c in s:
        out+=replacements.get(c,c)
        if c=='/': out+=r'\allowbreak{}'
    return out

def code(s):
    # Explicit discretionary breaks keep long C++ symbols and paths inside columns.
    out=''
    for i,c in enumerate(s):
        out+=esc(c)
        if c in '_:.' or (i and i%10==0): out+=r'\allowbreak{}'
    return r'{\ttfamily\small '+out+'}'

def inline(s):
    pattern=r'(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))'
    parts=re.split(pattern,s)
    out=''
    for part in parts:
        if part.startswith('`') and part.endswith('`'):out+=code(part[1:-1])
        elif part.startswith('**') and part.endswith('**'):out+=r'\textbf{'+inline(part[2:-2])+'}'
        elif part.startswith('[') and re.fullmatch(r'\[[^\]]+\]\([^)]+\)',part):
            m=re.match(r'\[([^\]]+)\]\(([^)]+)\)',part)
            out+=r'\href{'+m[2].replace('%',r'\%')+'}{'+esc(m[1])+'}'
        else:out+=esc(part)
    return out

lines=SRC.read_text().splitlines()
body=[]; i=next(i for i,line in enumerate(lines) if line.startswith('## '))
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('# '):i+=1;continue
    if line.startswith('```'):
        vals=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):vals.append(lines[i]);i+=1
        body.append('\\begin{lstlisting}\n'+'\n'.join(vals)+'\n\\end{lstlisting}\n');i+=1;continue
    if line.startswith('## '):
        body.append(r'\clearpage\section{'+esc(line[3:])+r'}'+'\n');i+=1;continue
    if line.startswith('### '):
        body.append(r'\subsection{'+esc(line[4:])+r'}'+'\n');i+=1;continue
    if line.startswith('| '):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',v.strip()) for v in row):rows.append(row)
            i+=1
        n=len(rows[0]);weights={2:[.21,.79],3:[.24,.38,.38],4:[.31,.12,.25,.32],5:[.075,.305,.265,.225,.13]}.get(n,[1/n]*n)
        if rows[0][0]=='Workload':weights=[.36,.10,.17,.19,.18]
        if rows[0][0]=='Code':weights=[.07,.93]
        cols=''.join(r'>{\raggedright\arraybackslash}p{'+str(round(w*6.50-.12,3))+'in}' for w in weights)
        tab=['{\\footnotesize\\setlength{\\tabcolsep}{4.3pt}\\renewcommand{\\arraystretch}{1.25}',r'\begin{longtable}{'+cols+'}',r'\toprule']
        header=' & '.join(r'\textbf{'+inline(v)+'}' for v in rows[0])+r' \\'
        tab += [header,r'\midrule\endfirsthead',header,r'\midrule\endhead',r'\bottomrule\endfoot']
        for row in rows[1:]:tab.append(' & '.join(inline(v) for v in row)+r' \\')
        tab += [r'\end{longtable}','}']
        body.append('\n'.join(tab)+'\n');continue
    if line.startswith('- '):
        vals=[]
        while i<len(lines) and lines[i].startswith('- '):vals.append(lines[i][2:]);i+=1
        body.append('\\begin{itemize}\n'+'\n'.join(r'\item '+inline(v) for v in vals)+'\n\\end{itemize}\n');continue
    vals=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','|','- ')):
        vals.append(lines[i]);i+=1
    body.append(inline(' '.join(vals))+'\n\n')

preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=22mm,headheight=15pt]{geometry}
\usepackage[table]{xcolor}
\usepackage{booktabs,longtable,array}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{hyperref}
\definecolor{navy}{HTML}{17324D}
\definecolor{teal}{HTML}{087F8C}
\definecolor{pale}{HTML}{F2F5F7}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal,pdftitle={RICH / MadVoro Performance Engineering Plan},pdfauthor={Codex code audit},pdfsubject={Runtime estimates, implementation contracts, and validation for distributed Voronoi construction}}
\setcounter{secnumdepth}{0}
\setcounter{tocdepth}{2}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{3em}
\setlist[itemize]{leftmargin=*,itemsep=3pt,topsep=4pt}
\titleformat{\section}{\Large\bfseries\color{navy}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\color{teal}}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{10pt}
\titlespacing*{\subsection}{0pt}{17pt}{7pt}
\lstset{basicstyle=\ttfamily\footnotesize,backgroundcolor=\color{pale},frame=single,rulecolor=\color{pale},breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true,showstringspaces=false,aboveskip=8pt,belowskip=8pt,xleftmargin=5pt,xrightmargin=5pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\color{navy} RICH / MadVoro}
\fancyhead[R]{\footnotesize Performance engineering plan}
\fancyfoot[L]{\scriptsize 7 September 2026 | Estimates are conditional, not measured gains}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\begin{document}
\begin{titlepage}
\thispagestyle{empty}
\vspace*{15mm}
{\color{teal}\large ENGINEERING AUDIT \& IMPLEMENTATION PLAN}\par
\vspace{12mm}
{\color{navy}\fontsize{32}{38}\selectfont\bfseries RICH / MadVoro}\par
\vspace{6mm}
{\color{navy}\fontsize{23}{29}\selectfont Faster distributed\\Voronoi construction\par}
\vspace{14mm}
{\large Concrete source targets, runtime contribution estimates,\\implementation contracts, and adversarial validation.}\par
\vspace{12mm}
{\color{teal}\rule{\textwidth}{1.5pt}}\par
\vspace{7mm}
\textbf{Primary recommendation}\par
Reduce avoidable MPI coordination and repeated preparation first.\par
Re-measure before changing triangulation or threading.\par
\vspace{9mm}
\textbf{Evidence}\par
Current-source audit across RICH and four dependencies.\par
Fresh 1 / 2 / 4-rank smoke runs and periodic validation.\par
24 numbered work items, including measurement prerequisites.\par
\vfill
\textbf{7 September 2026}\par
\small Source snapshot and raw evidence accompany this PDF.\par
All improvement ranges are planning estimates; no optimization speedup is claimed as measured.\par
\end{titlepage}
\tableofcontents
'''
OUT.write_text(preamble+'\n'.join(body)+'\n\\end{document}\n')
for passno in range(3):
    log=ROOT/f'pdf_build_{passno+1}.log'
    with log.open('w') as f:
        subprocess.run(['/usr/bin/pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(ROOT),str(OUT)],stdout=f,stderr=subprocess.STDOUT,check=True)
print(ROOT/'MadVoro_Performance_Plan.pdf')
