class SVGTransform:
    def __init__(self, shrink=1.0, w=500):
        self._shrink = shrink
        self._w = float(w)

        if self._shrink != 1.0:
            print "****WARN: shrinking sphere****"

    def xy_to_pxpy(self, x,y):
        w = self._w
        c = self._w / 2
        #center of svg is at 250,250 - move 0,0 there
        py = (x * +w * self._shrink) + c
        px = (y * -w * self._shrink) + c
        return px,py

    def pxpy_to_xy(self, px,py):
        w = self._w
        c = self._w / 2
        y = (px - c) / -w
        x = (py - c) / +w
        return x/self._shrink, y/self._shrink
