import re
import os
import pandas as pd
def readlog(filename):
    file = open(filename, "rU", encoding='utf-8').read()
    step = re.findall(r'step (.*?)', file)
    print(step)
if __name__ =='__main__':
    readlog('iphone.log')
#cat iphone.log | tail -1000000 | grep 'gen*' | tail -20
#cat a.txt  | awk -F ' ' '{print $5 $6}'
