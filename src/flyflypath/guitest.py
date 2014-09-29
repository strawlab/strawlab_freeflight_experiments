import sys

import model
import view
import euclid
import transform

XFORM = transform.SVGTransform()

from gi.repository import Gtk, GLib, Gdk

class TestView(view.SvgPathWidget):
    def __init__(self, model):
        view.SvgPathWidget.__init__(self, model)
        self.add_events(
                Gdk.EventMask.POINTER_MOTION_HINT_MASK | \
                Gdk.EventMask.POINTER_MOTION_MASK | \
                Gdk.EventMask.BUTTON_PRESS_MASK
        )
        self._mousex = self._mousey = None
        self.connect('motion-notify-event', self._on_motion_notify_event)
        self.connect('button-press-event', self._on_button_press_event)

    def _on_motion_notify_event(self, da, event):
        self._mousex = event.x
        self._mousey = event.y
        self.queue_draw()
        return True

    def _on_button_press_event(self, da, event):
        if event.type == Gdk.EventType.BUTTON_PRESS and event.button == 1:
            seg, ratio = self._model.connect_closest(p=None,px=event.x,py=event.y)
            self._model.start_move_from_ratio(ratio)
            return True

    def get_vec_and_points(self):
        vecs = []
        src_pt = None
        trg_pt = self._model.moving_pt if self._model else None
        if self._model and self._mousex is not None:
            src_pt = euclid.Point2(self._mousex,self._mousey)
            if self._model.moving_pt is not None:
                vecs.append((self._model.connect_to_moving_point(p=None,
                                    px=self._mousex,py=self._mousey),
                            (1,0,0))
                )

            seg, ratio = self._model.connect_closest(p=None,px=self._mousex,py=self._mousey)
            vecs.append((seg,(0,0,1)))

        return vecs,[(src_pt,(1,0,0)),(trg_pt,(0,1,0))],[]

    def get_annotations(self):
        return [(euclid.Point2(20,20),(0,1,0),"ratio: %.2f" % self.model.ratio)]

    def add_to_background(self, cr):

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (0.5)

        px,py = XFORM.xy_to_pxpy(0,0)
        cr.move_to(px,py)
        cr.show_text("0,0")

        cr.move_to(100,150)
        cr.show_text("(100,150)px")

        px,py = XFORM.xy_to_pxpy(0.2,0.4)
        cr.move_to(px,py)
        cr.show_text("(0.2,0.4)m")

        px,py = XFORM.xy_to_pxpy(-0.3,-0.1)
        cr.move_to(px,py)
        cr.show_text("(-0.3,-0.1)m")


        cr.stroke()


class Tester:
    def __init__(self, path):
        self._w = Gtk.Window()
        self._w.connect("delete-event", Gtk.main_quit)

        self._model = model.MovingPointSvgPath(path)
        self._view = TestView(self._model)

        self._model.start_move_from_ratio(0.3)

        self._w.add(self._view)
        self._w.show_all()

        self._step = 0.01
        GLib.timeout_add(200, self._move_along)

    def _move_along(self):
        self._model.advance_point(self._step, wrap=True)
        self._view.queue_draw()
        return True

if __name__ == "__main__":
    import os.path
    plain = os.path.join(os.path.dirname(__file__),'..','..','data','svgpaths','plain.svg')
    t = Tester(sys.argv[1] if len(sys.argv) > 1 else plain)
    Gtk.main()
