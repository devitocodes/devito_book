#!/bin/bash -x
# Compile the book to LaTeX/PDF.
#
# Usage: make.sh [nospell]
#
# With nospell, spellchecking is skipped.

set -x

name=book
topicname=fdm

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

rm tmp_* *.dlog

if [ $# -ge 1 ]; then
  spellcheck=$1
else
  spellcheck=spell
fi

if [ "$spellcheck" != 'nospell' ]; then
    system doconce spellcheck -d .dict4spell.txt book.do.txt preface.do.txt
    # No spellchecking of local files here since book.do.txt just includes files.
    # Spellcheck all *.do.txt files in each chapter.
    python -c 'import scripts; scripts.spellcheck()'
    if [ $? -ne 0 ]; then
	echo "Go to relevant directory, run bash make.sh and update dictionary!"
	exit 1
    fi
fi

cp ../chapters/newcommands_keep.p.tex newcommands_keep.tex
doconce replace 'newcommand{\E}' 'renewcommand{\E}' newcommands_keep.tex
doconce replace 'newcommand{\I}' 'renewcommand{\I}' newcommands_keep.tex

opt1="DOCUMENT=book --encoding=utf-8"

function edit_solution_admons {
    # We use question admon for typesetting solution, but let's edit to
    # somewhat less eye catching than the std admon
    # (also note we use --latex_admon_envir_map= in compile)
    doconce replace 'notice_mdfboxadmon}[Solution.]' 'question_mdfboxadmon}[Solution.]' ${name}.tex
    doconce replace 'end{notice_mdfboxadmon} % title: Solution.' 'end{question_mdfboxadmon} % title: Solution.' ${name}.tex
    doconce subst -s '% "question" admon.+?question_mdfboxmdframed\}' '% "question" admon
\\colorlet{mdfbox_question_background}{gray!5}
\\newmdenv[        % edited for solution admons in exercises
  skipabove=15pt,
  skipbelow=15pt,
  outerlinewidth=0,
  backgroundcolor=white,
  linecolor=black,
  linewidth=1pt,       % frame thickness
  frametitlebackgroundcolor=blue!5,
  frametitlerule=true,
  frametitlefont=\\normalfont\\bfseries,
  shadow=false,        % frame shadow?
  shadowsize=11pt,
  leftmargin=0,
  rightmargin=0,
  roundcorner=5,
  needspace=0pt,
]{question_mdfboxmdframed}' ${name}.tex
}

function compile {
    options="$@"
system doconce format pdflatex $name $opt1 --exercise_numbering=chapter --exercise_solution=admon --latex_admon_envir_map=2 --latex_style=Springer_T4 --latex_title_layout=titlepage --latex_list_of_exercises=loe --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_no_program_footnotelink --latex_copyright=titlepages "--latex_code_style=default:lst[style=blue1_bluegreen]@pypro:lst[style=blue1bar_bluegreen]@pypro2:lst[style=greenblue]@pycod2:lst[style=greenblue]@dat:lst[style=gray]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --movie_prefix=https://raw.githubusercontent.com/hplgit/fdm-book/master/doc/.src/book/ --latex_list_of_exercises=toc $options

# Auto edits
edit_solution_admons
# Post-process generated LaTeX file to fix known issues.
NAME=$name python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["NAME"] + ".tex")
text = path.read_text()

# Fix 1: Replace title block with minimal version to avoid nested center environments.
start = text.find("% ----------------- title -------------------------")
end = text.find(r"\clearpage", start) if start != -1 else -1

if start != -1 and end != -1:
    head = text[:start]
    tail = text[end:]
    minimal_title = r"""% ----------------- title -------------------------
\thispagestyle{empty}
\hbox{\ \ }
\vfill
\begin{center}
{\Huge\bfseries Finite Difference Computing with PDEs - A Modern Software Approach\par}
\vspace{6mm}
{\Large Hans Petter Langtangen\par}
{\Large Svein Linge\par}
\vspace{6mm}
{\large Jan 23, 2026}
\end{center}
\vfill
"""
    text = head + minimal_title + tail

# Fix 2: Change from utf8x (with ucs package) to standard utf8 encoding.
# The utf8x/ucs combination causes "Argument of X has an extra }" errors.
text = text.replace(r"\usepackage{ucs}", r"%\usepackage{ucs}  % DISABLED: causes encoding issues with utf8x")
text = text.replace(r"\usepackage[utf8x]{inputenc}", r"\usepackage[utf8]{inputenc}  % Changed from utf8x to utf8")

# Fix 3: Add Cython language definition for listings package.
# DocOnce generates language=cython for .pyx files, but listings doesn't have cython built-in.
cython_def = r'''
% Cython language definition for listings (Cython is a superset of Python)
\lstdefinelanguage{cython}[]{python}{
  morekeywords={cdef,cpdef,ctypedef,cimport,DEF,IF,ELIF,ELSE,nogil,gil,with,extern,namespace,fused,readonly,public,api,inline,bint,Py_ssize_t},
}
'''
# Insert after \begin{document}
text = text.replace(r"\begin{document}", cython_def + r"\begin{document}")

path.write_text(text)
PY
# With t4/svmono linewidth has some too large value before \mymainmatter
# is called, so the box width as linewidth+2mm is wrong, it must be
# explicitly set to 120mm.
doconce replace '\setlength{\lstboxwidth}{\linewidth+2mm}' '\setlength{\lstboxwidth}{120mm}' $name.tex  # lst
system doconce replace 'linecolor=black,' 'linecolor=darkblue,' $name.tex
system doconce subst 'frametitlebackgroundcolor=.*?,' 'frametitlebackgroundcolor=blue!5,' $name.tex
system doconce replace '\maketitle' '\subtitle{Modeling, Algorithms, Analysis, Programming, and Verification}\maketitle' $name.tex

rm -rf $name.aux $name.ind $name.idx $name.bbl $name.toc $name.loe

system pdflatex $name
system bibtex $name
system makeindex $name
system pdflatex $name
system pdflatex $name
system makeindex $name
system pdflatex $name
}

# Compile with solutions for screen
compile --device=screen --skip_inline_comments

# Report typical problems (lines more than 10pt too long)
doconce latex_problems $name.log 10

# Publish
dest=../../../doc/pub/book/pdf
mkdir -p $dest
cp $name.pdf $dest/${topicname}-book.pdf
