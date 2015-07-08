class SVGTransform:
    def __init__(self, shrink=1.0, w=500):
        self._shrink = shrink
        self._w = float(w)

        if self._shrink != 1.0:
            print "****WARN: shrinking sphere****"

    @property
    def size_px(self):
        return (self._w,self._w)

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

    def xyz_to_pxpypz(self,x,y,z):
        px,py = self.xy_to_pxpy(x,y)
        return px,py,z

    def pxpypz_to_xyz(self,px,py,pz):
        x,y = self.pxpy_to_xy(px,py)
        return x,y,pz



