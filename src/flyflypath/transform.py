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
        py = (y * -w * self._shrink) + c
        px = (x * +w * self._shrink) + c
        return px,py

    def pxpy_to_xy(self, px,py):
        w = self._w
        c = self._w / 2
        y = (py - c) / -w
        x = (px - c) / +w
        return x/self._shrink, y/self._shrink

    def m_to_pixel(self, m):
        return m*self._w*self._shrink

    def pixel_to_m(self, m):
        return m/self._w/self._shrink


if __name__ == "__main__":
    XFORM = SVGTransform()

    xy = [(0,0),(0.2,0.4),(-0.3,-0.1)]
    for x,y in xy:
        px,py = XFORM.xy_to_pxpy(x,y)
        _x,_y = XFORM.pxpy_to_xy(px,py)
        assert x == _x
        assert y == _y

    m = XFORM.pixel_to_m(20)
    p = XFORM.m_to_pixel(m)

    assert p == 20

