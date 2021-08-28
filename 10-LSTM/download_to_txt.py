for i in ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017']:
    !wget 'https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_{i}.json'
    !mv 'dialogues_{i}.json' '../../data/multiwoz/'

import json
import pandas as pd
  
for i in ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017']:
    if i =='001':
        f = open(f'../../data/multiwoz/dialogues_{i}.json')
        data = json.load(f)
        all_question, all_answer = [], []

        for qa in data:
            all_question.append(qa[u'turns'][0][u'utterance'])
            all_answer.append(qa[u'turns'][1][u'utterance'])

            all_ = pd.DataFrame([all_question,all_answer]).T

    else : 
        f = open(f'../../data/multiwoz/dialogues_{i}.json')
        data = json.load(f)
        all_question, all_answer = [], []

        for qa in data:
            all_question.append(qa[u'turns'][0][u'utterance'])
            all_answer.append(qa[u'turns'][1][u'utterance'])

        new = pd.DataFrame([all_question,all_answer]).T

        all_ = pd.concat([all_, new], axis=0)

import numpy as np

np.savetxt(r'../../data/hotpotqa/multiwoz_all.txt', all.values, fmt='%s', delimiter='\t')