#!/bin/bash

# To be run from one directory above this script.

. ./path.sh

stage=1

. utils/parse_options.sh

cd corpora/wikimedia

if [ $stage -le 1 ]; then
    wget -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
    git clone https://github.com/attardi/wikiextractor.git wikiextractor
    python3 wikiextractor/WikiExtractor.py -o enwiki -b100M -c --no_templates --filter_disambig_pages enwiki-latest-pages-articles-multistream.xml.bz2
fi

# sudo apt install cmake libboost-all-dev libeigen3-dev

if [ $stage -le 2 ]; then
    alias egrep='egrep --line-buffered'
    wikifind() {
        find "$1" -type f | sort -n
    }
    wikicat() {
        wikifind "$1" | xargs bzcat
    }
    wikifilter() {
        egrep -v '[<>|0-9\-]' |\
            egrep '\.$' |\
            # tr '[A-Z]' '[a-z]' |\
            tr -dC "[ A-Za-z'.\n]" |\
            sed -Ee 's,\([^)]*\),,g' -e 's/ \.\.\. / /g' |\
            sed -Ee "s/'([A-Za-z]*)'/\\1/g"
    }
    extract_vocab() {
        egrep -o "\\b[a-z0-9'\\-]*\\b"
    }

    if [[ ! -e "lexicon" ]]; then
        echo "[+] Extracting lexicon"
        IFS=$'\n'
        :>lexicon
        for file in $(wikifind enwiki); do
            echo " [-] $file"
            bzcat "$file" | wikifilter | extract_vocab | sort -u >> lexicon
        done
    fi

    # 3 4 5 6
    for n in 4; do
        name=enwiki-${n}gram
        # -S 4G -o 4 --prune 0 1 2
        wikicat "enwiki" | wikifilter | ~/build/kenlm/build/bin/lmplz -o $n --prune 0 3 5 7 --temp_prefix . > $name.arpa
        xz -v -f -k $name.arpa
        ~/build/kenlm/build/bin/build_binary $name.arpa $name.bin
        xz -v -f $name.bin
        # arpa2fst --disambig-symbol=\#0 --read-symbol-table=words.txt $name.arpa $name.fst
    done
fi
