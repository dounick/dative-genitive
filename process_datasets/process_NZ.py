import pandas as pd

dat = pd.read_csv('NZdatives.csv')
dat.fillna('', inplace=True)
dat = dat[['Line','Construction', 'verb.inflected', 'rec', 'rec.def', 'rec.pron', 'rec.anim', 'thm', 'thm.def', 'thm.pron', 'thm.anim']]
dat.rename(columns={'verb.inflected':'verb'}, inplace=True)
dat.to_csv('NZ_dat.csv', index=False)

gen = pd.read_csv('NZgenitives_nopron.csv')
gen.fillna('', inplace=True)
gen = gen[['Line', 'Construction', 'type', 'psr', 'psr-def', 'psr-pron', 'psr-anim.bin', 'psm', 'psm-pron', 'psm.anim.bin']]
gen.rename(columns={'psr-def' : 'psr.def', 
                    'psr-pron' : 'psr.pron',
                     'psr-anim.bin' : 'psr.anim', 
                    'psm-pron':'psm.pron', 
                    'psm.anim.bin':'psm.anim'}, inplace=True)
gen.to_csv('NZ_gen.csv', index=False)

