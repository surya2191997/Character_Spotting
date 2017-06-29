# Script for splitting train and test

import os
import shutil
import random

def decision(probability):
    return random.random() > probability

validation_split=0.9
dstdir=dstroot = '/usr2/prouserdata/surya/TrackBali/glyph_train'
i=0
for fname in os.listdir('/usr2/prouserdata/surya/TrackBali/glyph_test'):
	i=i+1
        #srcfile='/usr2/prouserdata/surya/TrackBali/glyph_test/'+fname
	#if decision(validation_split) :
		#shutil.move(srcfile, dstdir)

print(i)


###  SCRIPT TO MOVE 30% DATA TO TEST_FOLDER ###
