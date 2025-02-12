#!/bin/bash

clean() {
    rm -f main.fls
    rm -f main.fdb_latexmk
    rm -f main.aux
    rm -f main.log
    rm -f main.out
    rm -f main.pdf
}

latex() {
    clean
    latexmk -pdf -shell-escape main.tex
}

if [ "$1" == "clean" ]; then
  clean
elif [ "$1" == "latex" ]; then
  latex
else
  echo "Usage: ./make clean | ./make latex"
  exit 1
fi
