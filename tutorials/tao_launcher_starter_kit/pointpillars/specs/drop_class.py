# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

import os
import sys


def drop_class(label_dir, classes):
    """drop label by class names."""
    labels = os.listdir(label_dir)
    labels = [os.path.join(label_dir, x) for x in labels]
    for gt in labels:
        print("Processing ", gt)
        with open(gt) as f:
            lines = f.readlines()
            lines_ret = []
            for line in lines:
                ls = line.strip()
                line = ls.split()
                if line[0] in classes:
                    print("Dropping ", line[0])
                    continue
                else:
                    lines_ret.append(ls)
        with open(gt, "w") as fo:
            out = '\n'.join(lines_ret)
            fo.write(out)


if __name__ == "__main__":
    drop_class(sys.argv[1], sys.argv[2].split(','))
