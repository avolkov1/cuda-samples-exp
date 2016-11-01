#!/bin/bash

rebuild=false

case $1 in
    -r|--rebuild)
        rebuild=true
        shift
        ;;
esac

build_target=cdpSimplePrint
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
buildir=${codedir}/build

mkdir -p ${buildir}

cd ${buildir}

if [ "$rebuild" = true ] ; then
    rm -r ${buildir}/*
fi

cmake ${codedir}
make

${buildir}/cdpSimplePrint
