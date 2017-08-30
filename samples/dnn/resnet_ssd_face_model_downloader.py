#!/usr/bin/env python
from __future__ import print_function

if __name__ == '__main__':
    print("This is not a standalone sample.")
    sys.exit(1)


import hashlib
import sys
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


class Model:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.url = kwargs.pop('url')
        self.filename = kwargs.pop('filename')
        self.md5 = kwargs.pop('md5', None)

    def __str__(self):
        return 'Model <{}>'.format(self.name)

    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('  {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

    def verify(self):
        if not self.md5:
            return False
        print('  expect {}'.format(m.md5))
        hash = hashlib.md5()
        with open(self.filename, 'rb') as f:
            while True:
                buf = f.read(self.BUFSIZE)
                if not buf:
                    break
                hash.update(buf)
        print('  actual {}'.format(hash.hexdigest()))
        return self.md5 == hash.hexdigest()

    def download(self):
        try:
            if self.verify():
                print('  hash match - skipping download')
                return
        except Exception as e:
            print('  catch {}'.format(e))
        print('  hash check failed - downloading')
        print('  get {}'.format(self.url))
        r = urlopen(self.url)
        self.printRequest(r)
        with open(self.filename, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()
            print(' done')
        print(' file {}'.format(self.filename))
        self.verify()


m = Model(
    name='FaceDetectorModel',
    url='https://raw.githubusercontent.com/opencv/opencv_3rdparty/b2bfc75f6aea5b1f834ff0f0b865a7c18ff1459f/res10_300x300_ssd_iter_140000.caffemodel',
    md5='afbb6037fd180e8d2acb3b58ca737b9e',
    filename='face_detector/res10_300x300_ssd_iter_140000.caffemodel')
m.download()
