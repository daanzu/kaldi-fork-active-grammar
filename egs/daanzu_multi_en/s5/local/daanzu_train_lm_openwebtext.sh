#!/bin/bash

stage=1

if [ $stage -le 1 ]; then
    # git clone https://github.com/chentinghao/download_google_drive.git
    # python3 download_google_drive/download_gdrive.py 1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx openwebtext.tar.xz
    # tar Jxvf openwebtext.tar.xz
    :
fi

if [ $stage -le 2 ]; then
    alias egrep='egrep --line-buffered'
    owt_find() {
        find "$1" -type f | sort -n
    }
    owt_cat() {
        owt_find "$1" | xargs -n1 tar JxOf
    }
    owt_filter() {
        # blank lines
        awk NF |\
        # acronyms
        sed -Ere "s/\b([A-Z]){2,}\b('s)?//g" |\
        sed -Ere "s/\b([A-Z]\.){2,}\B('s)?//g" |\
        sed -Ere "s/\b([a-z]\.){2,}\B('s)?//g" |\
        # convert to lowercase
        tr '[A-Z]' '[a-z]' |\
        # curly quotes: ‘weird’ “I move slowly on my bike builds,”
        sed -Ere "s/\B‘([^’]*)’\B/\\1/g" |\
        tr '’' "'" |\
        sed -Ere "s/'''/'/g" -Ere "s/''//g" |\
        # single-quoted-started and -ended words
        sed -Ere "s/\B'([A-Za-z]*)/\\1/g" |\
        sed -Ere "s/([A-Za-z]*)'\B/\\1/g" |\
        # em dashes
        sed -Ere "s/\B-{2,}\B//g" |\
        # remove all other characters
        tr -dC "[ A-Za-z'\-\n]" |\
        # extraneous parens, brackets, etc
        sed -Ere 's/\([^)]*\)//g' -e 's/\[[^]]*\]//g' -e 's/ \.\.\. / /g' #|\
        # unquote single quoted started expressions
        # sed -Ere "s/'([A-Za-z]*)'/\\1/g"
        # egrep -v '[<>|0-9\-]' |\
        #     egrep '\.$' |\
        #     # tr '[A-Z]' '[a-z]' |\
        #     tr -dC "[ A-Za-z'.\n]" |\
        #     sed -Ee 's,\([^)]*\),,g' -e 's/ \.\.\. / /g' |\
        #     sed -Ee "s/'([A-Za-z]*)'/\\1/g"
    }
    extract_vocab() {
        egrep -o "\\b[a-z0-9'\\-]*\\b"
    }

    # owt_cat openwebtext/
    # owt_cat openwebtext/ | owt_filter
    # owt_cat openwebtext/ | owt_filter | wc
    # exit 0

    # if [[ ! -e "lexicon.txt" ]]; then
    #     echo "[+] Extracting lexicon.txt"
    #     IFS=$'\n'
    #     :>lexicon.txt
    #     for file in $(owt_find openwebtext/); do
    #         echo " [-] $file"
    #         bzcat "$file" | wikifilter | extract_vocab | sort -u >> lexicon.txt
    #     done
    # fi

    # 3 4 5 6
    for n in 3; do
        name=openwebtext-${n}gram-0-32-64
        # -S 4G -o 4 --prune 0 1 2
        owt_cat "openwebtext/" | owt_filter | ~/build/kenlm/build/bin/lmplz -o $n --prune 0 32 64 --temp_prefix . > $name.arpa
        xz -v -f -k $name.arpa
        # ~/build/kenlm/build/bin/build_binary $name.arpa $name.bin
        # xz -v -f $name.bin
        # arpa2fst --disambig-symbol=\#0 --read-symbol-table=words.txt $name.arpa $name.fst
    done
fi
