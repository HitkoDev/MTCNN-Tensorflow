"""Convert AWE data with bitmasks to txt annotations"""

# MIT License
#
# Copyright (c) 2017 HitkoDev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil

import cv2
from glob import glob


def main():
    awe_dir = 'AWEForSegmentation'
    folders = [x for x in os.listdir(awe_dir)]
    folders = [
        [
            f,
            '{}annot_rect'.format(f)
        ]
        for f in folders if '{}annot_rect'.format(f) in folders
    ]
    for folder, annot in folders:
        save_dir = 'AWE_{}'.format(folder)
        src_dir = os.path.join(awe_dir, folder)
        annot_dir = os.path.join(awe_dir, annot)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        images = glob('{}/**/*'.format(src_dir), recursive=True)
        images = [i for i in images if os.path.exists(i.replace(src_dir, annot_dir))]
        with open('{}.txt'.format(save_dir), 'w+') as file:
            for path in images:
                shutil.copy2(path, path.replace(src_dir, save_dir))
                mask = cv2.imread(path.replace(src_dir, annot_dir))
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                data = [path.replace(src_dir, '')[1:]]
                for c in contours:
                    x1 = min([p[0][0] for p in c])
                    x2 = max([p[0][0] for p in c])
                    y1 = min([p[0][1] for p in c])
                    y2 = max([p[0][1] for p in c])
                    data.append(' '.join(str(x) for x in [x1, y1, x2, y2]))
                file.write(' '.join(data))
                file.write('\n')


if __name__ == '__main__':
    main()
