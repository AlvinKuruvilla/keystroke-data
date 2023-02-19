#!/bin/bash
arr=(/Users/alvinkuruvilla/Dev/orl-keystroke/raw_data/*/) # This creates an array of the full paths to all subdirs
arr=("${arr[@]%/}")                                       # This removes the trailing slash on each item
arr=("${arr[@]##*/}")                                     # This removes the path prefix, leaving just the dir names
echo "${arr[*]}"
declare -i prefix=11
for i in "${arr[@]}"; do
    :
    cd /Users/alvinkuruvilla/Dev/orl-keystroke/raw_data/$i/ || exit
    ls | xargs -I {} mv {} $prefix{}
    cd ..
    ((prefix = prefix + 1))
done
