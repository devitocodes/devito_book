#!/bin/bash
# Generic make.sh for building PDF chapters
# Usage: make.sh name

set -x

nickname=$1
mainname=main_$nickname

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

if [ $# -eq 0 ]; then
  echo 'name of document missing!'
  exit 1
fi

rm -f tmp_*

doconce spellcheck -d .dict4spell.txt *.do.txt
if [ $? -ne 0 ]; then
  echo "make.sh aborts due to misspellings"
  exit 1
fi
rm -rf tmp_stripped*

egrep "[^\\]thinspace" *.do.txt
if [ $? -eq 0 ]; then echo "wrong thinspace commands - abort"; exit; fi

comments="--skip_inline_comments"
doc=document
appendix=document

preprocessor_opt="DOCUMENT=$doc APPENDIX=$appendix BOOK=standalone FEM_BOOK=False -DNOTREAD"
no_solutions='--without_solutions --without_answers'

rm -f *.aux
preprocess -DFORMAT=pdflatex ../newcommands_keep.p.tex > newcommands_keep.tex

function edit_solution_admons {
    # We use question admon for typesetting solution, but let's edit to
    # somewhat less eye catching than the std admon
    doconce replace 'notice_mdfboxadmon}[Solution.]' 'question_mdfboxadmon}[Solution.]' ${mainname}.tex
    doconce replace 'end{notice_mdfboxadmon} % title: Solution.' 'end{question_mdfboxadmon} % title: Solution.' ${mainname}.tex
    doconce subst -s '% "question" admon.+?question_mdfboxmdframed\}' '% "question" admon
\colorlet{mdfbox_question_background}{gray!5}
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
]{question_mdfboxmdframed}' ${mainname}.tex
}

function compile {
    options="$@"
    system doconce format pdflatex ${mainname} $preprocessor_opt $comments \
        --latex_table_format=center \
        "--latex_code_style=default:lst[style=blue1_bluegreen]@pypro:lst[style=blue1bar_bluegreen]@pypro2:lst[style=greenblue]@pycod2:lst[style=greenblue]@dat:lst[style=gray]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" \
        --allow_refs_to_external_docs \
        --exercise_solution=admon \
        --movie_prefix=https://raw.githubusercontent.com/devitocodes/devito_book/master/doc/.src/chapters/${nickname}/ \
        --latex_admon_envir_map=2 \
        --latex_list_of_exercises=toc \
        $options
    edit_solution_admons
    system doconce latex_exercise_toc ${mainname}
    doconce subst 'frametitlebackgroundcolor=.*?,' 'frametitlebackgroundcolor=blue!5,' ${mainname}.tex
    system pdflatex ${mainname}
    system makeindex ${mainname}
    system bibtex ${mainname}
    pdflatex ${mainname}
    pdflatex ${mainname}
}

# PDF with solutions
compile --device=screen
cp ${mainname}.pdf ${nickname}-sol.pdf
rm -f ${mainname}.pdf

# PDF for printing
compile --device=paper $no_solutions
cp ${mainname}.pdf ${nickname}-4print.pdf

# PDF for screen viewing
compile --device=screen $no_solutions
cp ${mainname}.pdf ${nickname}-4screen.pdf
rm -f ${mainname}.pdf

echo "Generated PDFs:"
ls -la ${nickname}*.pdf
