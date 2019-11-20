#!/usr/bin/env python3

import sys
import glob
import sox

total = 0

for path in glob.iglob(sys.argv[1] + '/**/*', recursive=True):
    # print(path)
    if any(path.endswith(ext) for ext in '.wav .mp3'.split()):
        try:
            duration = sox.file_info.duration(path)
            print("%s  %.3f" % (path, duration))
            total += duration
        except Exception as e:
            pass

print("TOTAL  %.3f" % (total))
