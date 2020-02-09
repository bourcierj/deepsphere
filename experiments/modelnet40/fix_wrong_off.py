import sys
import os
import glob
import re
from tqdm import tqdm
"""
Correct OFF files with wrong headers
There are files with header line like: OFF18515 26870 0
with the counts attached to 'OFF'
Trimesh raises an exception with those files. Correct them by separating the counts
and writing them on the next line, like the sane files.
In total : 2862 corrected files.
"""

def _collect_off_files(path):
    files = glob.glob(os.path.join(path, "**/*.off"), recursive=True)
    return files


if __name__ == '__main__':

    path = './data/ModelNet40'
    verbose = True
    files = _collect_off_files(path)
    if len(files) == 0:
        print('No OFF files in directory.')
        sys.exit(1)

    wrong_files = []
    wrong = False
    for file in (tqdm(files) if verbose else files):
        with open(file, 'r') as file_obj:
            text = file_obj.read()

            text = text.lstrip()
            # split the first line
            header, raw = text.split('\n', 1)
            filename = '/'.join(file.split('/')[-3:])
            if header.upper() not in ['OFF', 'COFF']:
                print('{}: Wrong header: {}  :'.format(filename, header), end=' ')
                wrong = True
                wrong_files.append(file)
        if wrong:
            splitted = re.split(r'(\d+)', header)
            header = splitted[0]
            counts = ''.join(splitted[1:])
            with open(file, 'w') as file_obj:
                # write corrected file
                file_obj.write(header+'\n'+counts+'\n'+raw)
                print('corrected')
            wrong = False

    print('Total number of files with corrected: {}'.format(len(wrong_files)))

    # Assert that the files are now correctly read with Trimesh
    # import trimesh
    # for file in tqdm(files):
    #     try:
    #         mesh = trimesh.load_mesh(file)
    #     except NameError:
    #         filename = '/'.join(file.split('/')[-3:])
    #         print('{}: Wrong header: {}'.format(filename, header))
    #         raise
