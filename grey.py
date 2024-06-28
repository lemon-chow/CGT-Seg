import os
import glob
import cv2

def togrey(img,outdir):
    src = cv2.imread(img) 
    try:
        dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(outdir,os.path.basename(img)), dst)
    except Exception as e:
        print(e)

for file in glob.glob('data_0/imgs/5/88.png'):  
    togrey(file,'grey')
