import os
import glob


img_root = 'datasets/ICPR/ICPR_train/image'
txt_root = 'datasets/ICPR/ICPR_train/txt'

imgs = glob.glob(os.path.join(img_root, '*.jpg'))


a = []
for im in imgs:
    f = os.path.splitext(os.path.basename(im))[0] + '.txt'
    t = os.path.join(txt_root, f)
    if os.path.exists(t):
        one = '{}   {}\n'.format(im, t)
        a.append(one)

with open('datasets/icpr_train.txt', 'w') as f:
    f.writelines(a)

print('done.')