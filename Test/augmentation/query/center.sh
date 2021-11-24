#!/bin/bash
parallel -vk  ::: \
'python center.py --num 0' \
'python center.py --num 1' \
'python center.py --num 2' \
'python center.py --num 3' \
'python center.py --num 4' \
'python center.py --num 5' \
'python center.py --num 6' \
'python center.py --num 7' \
'python center.py --num 8' \
'python center.py --num 9' \
'python center.py --num 10' \
'python center.py --num 11' \
'python center.py --num 12' \
'python center.py --num 13' \
'python center.py --num 14' \
'python center.py --num 15' \
'python center.py --num 16' \
'python center.py --num 17' \
'python center.py --num 18' \
'python center.py --num 19'
