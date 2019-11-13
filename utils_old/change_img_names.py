import os

dir = "/Users/ivan/projects/set/data/raw/out"

map = {'0': 'p3',
       '1': 'p2',
       '2': 'p1',
       '3': 'r3',
       '4': 'r2',
       '5': 'r1',
       '6': 'g3',
       '7': 'g2',
       '8': 'g1',
       '9': '_all'
       }

for fi in os.listdir(dir):
    if os.path.splitext(fi)[-1] == '.png':
        name = os.path.basename(fi).split('.')[0]
        basename, number = name.split('_')
        newname = basename + map[number] + '.png'
        oldpath = os.path.join(dir, fi)
        newpath = os.path.join(dir, newname)
        os.rename(oldpath, newpath)
        print(oldpath)
        print('renamed')
        print(newpath)
        
            
    


