import pandas as pd
import util

# Process each sentence
dat = pd.read_csv('NZ_dat_original.csv')
dat.fillna('', inplace=True)

for index, row in dat.iterrows():
    # Get the words from each row
    rec = row['rec']
    thm = row['thm']
    verb = row['verb']
    construction = row['Construction']
    sentence = dat.loc[index, 'Line']
    modified_sentence = util.modify_dative_sentence(sentence, construction, verb, rec, thm)
    if modified_sentence != -1:
        dat.loc[index, 'Line'] = modified_sentence
    else:
        dat.drop(index, inplace=True)


dat.to_csv('NZ_dat_attempt.csv', index=False)
# Print the processed sentences
