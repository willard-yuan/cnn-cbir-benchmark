
import scipy.io
mat = scipy.io.loadmat('mat_temp.mat')

X = mat['X']

ovr = 0.4
steps = [2,3,4,5,6,7]

W = X.shape[1]
H = X.shape[0]

w = min(W, H)

import math
w2 = math.floor(w/2 -1)

b = [(max(H, W)-w)*1.0/(step-1) for step in steps]

test = [abs((w*w - w*bi)/(w*w) - ovr) for bi in b]

idx = test.index(min(test)) + 1

Wd = 0
Hd = 0

if H < W:
    Wd = idx
elif H > W:
    Hd = idx

wl = math.floor(2*w/(l+1))
wl2 = math.floor(wl/2 - 1)


if (l+Wd-1) == 0:
    b = 0
else:
    b = (W-wl)/(l+Wd-1)

cenW = [math.floor(wl2 + tmp_i*b) -wl2 for tmp_i in xrange(l-1+Wd+1)]


if (l+Hd-1) == 0:
    b = 0
else:
    b = (H-wl)/(l+Hd-1)

cenW = [math.floor(wl2 + tmp_i*b) -wl2 for tmp_i in xrange(l-1+Hd+1)]


R = X[0:int(wl)][0:int(wl)]

if not bool(min(R.shape)):
    continue
