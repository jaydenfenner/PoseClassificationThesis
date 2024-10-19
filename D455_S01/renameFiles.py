'''
For renaming files in D455_S01

Naming convention:
f"S01_{side}_{cover}_{pillow}"
- side in: ['B', 'R', 'F', 'LH']
- cover in ['0', 'S-60', 'S-90', 'Q-60', 'Q-90']
- pillow in: ['wp', 'np']

NUMBERING:
===========================================================
    BACK AND RIGHT
    - 1: no pillow, no occlusion

    - 2: no pillow, 60% sheet
    - 3: no pillow, 90% sheet
    --------------------------------------
    - 4: no pillow, 60% quilt
    - 5: no pillow, 90% quilt
    --------------------------------------
    - 6: pillow, no occlusion

    - 7: pillow, 60% sheet
    - 8: pillow, 90% sheet

    - 9: pillow, 60% quilt
    - 10: pillow, 90% quilt
    --------------------------------------
===========================================================
    FRONT AND LEFT_(HARD)
    - 1: no pillow, no occlusion
    - 2: pillow, no occlusion

    - 3: no pillow, 60% sheet
    - 4: pillow, 60% sheet
    - 5: no pillow, 90% sheet
    - 6: pillow, 90% sheet
    --------------------------------------
    - 7: no pillow, 60% quilt
    - 8: pillow, 60% quilt
    - 9: no pillow, 90% quilt
    - 10: pillow, 90% quilt
    --------------------------------------
===========================================================
'''

import os

def rename_files():
    basePath = 'D455_S01' # ralative path to session folder from root

    numbers = range(1, 11)
    suffixes = ['Depth.png', 'Depth.raw', 'Depth_metadata.csv', 'Color.png']
    files = [] # [filename, suffix]
    for number in numbers:
        for suffix in suffixes:
            files.append([f"{number}_{suffix}", suffix])

    poseTuples = [
        ('BACK', 'B'),
        ('RIGHT', 'R'),
        ('FRONT', 'F'),
        ('LEFT_(HARD)', 'LH'),
    ]
    covers = ['0', 'S-60', 'S-90', 'Q-60', 'Q-90']
    pillows = ['np', 'wp']

    for poseTuple in poseTuples:
        poseFolderPath = os.path.join(basePath, poseTuple[0])

        for number in range(1, 11): # (1 through 10)
                
            if (poseTuple[0] == 'BACK' or poseTuple[0] == 'RIGHT'): #! NOTE all covers done before changing pillow
                pillow = pillows[(number-1) // 5]
                cover = covers[(number-1) % 5]

                for suffix in suffixes:
                    originalFilename = f"{number}_{suffix}"
                    originalPath = os.path.join(basePath, poseTuple[0], originalFilename)
                    newFileName = f"S01_{poseTuple[1]}_{cover}_{pillow}_{suffix}"
                    newPath = os.path.join(basePath, poseTuple[0], newFileName)
                    os.rename(originalPath, newPath)
                    if suffix == suffixes[1]:
                        print(f"renamed {poseTuple[0]}: {originalFilename} -> {newFileName}")

            elif (poseTuple[0] == 'FRONT' or poseTuple[0] == 'LEFT_(HARD)'): #! NOTE both pillows done before changing cover
                pillow = pillows[(number-1) % 2]
                cover = covers[(number-1) // 2]

                for suffix in suffixes:
                    originalFilename = f"{number}_{suffix}"
                    originalPath = os.path.join(basePath, poseTuple[0], originalFilename)
                    newFileName = f"S01_{poseTuple[1]}_{cover}_{pillow}_{suffix}"
                    newPath = os.path.join(basePath, poseTuple[0], newFileName)
                    os.rename(originalPath, newPath)
                    if suffix == suffixes[1]:
                        print(f"renamed {poseTuple[0]}: {originalFilename} -> {newFileName}")

if __name__ == "__main__":
    # rename_files()
    print('This script has already been run')