import re
import pandas as pd
import util

#print(util.modify_genitive_sentence('this is the king of britain', 'britain', 'king', 'OF'))
#print(util.modify_genitive_sentence('this is britain\'s king', 'britain', 'king', 'S'))
# Process each sentence

gen = pd.read_csv('NZ_gen_original.csv')
gen.fillna('', inplace=True)

for index, row in gen.iterrows():
    # Get the words from each row
    type = row['type']
    psr = row['psr']
    psm = row['psm']
    sentence = gen.loc[index, 'Line']
    modified_sentence = util.modify_genitive_sentence(sentence, psr, psm, type)
    if modified_sentence != -1:
        gen.loc[index, 'Line'] = modified_sentence
    else:
        gen.drop(index, inplace=True)


gen.to_csv('NZ_gen_attempt.csv', index=False)
# Print the processed sentences'''
