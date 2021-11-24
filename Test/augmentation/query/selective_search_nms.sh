#!/bin/bash

parallel -vk  ::: \
'python selective_search_nms.py --num 0' \
'python selective_search_nms.py --num 1' \
'python selective_search_nms.py --num 2' \
'python selective_search_nms.py --num 3' \
'python selective_search_nms.py --num 4' \
'python selective_search_nms.py --num 5' \
'python selective_search_nms.py --num 6' \
'python selective_search_nms.py --num 7' \
'python selective_search_nms.py --num 8' \
'python selective_search_nms.py --num 9' \
'python selective_search_nms.py --num 10' \
'python selective_search_nms.py --num 11' \
'python selective_search_nms.py --num 12' \
'python selective_search_nms.py --num 13' \
'python selective_search_nms.py --num 14' \
'python selective_search_nms.py --num 15' \
'python selective_search_nms.py --num 16' \
'python selective_search_nms.py --num 17' \
'python selective_search_nms.py --num 18' \
'python selective_search_nms.py --num 19'
