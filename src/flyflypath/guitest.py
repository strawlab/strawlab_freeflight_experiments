import sys

import model
import view
import euclid
import transform

XFORM = transform.SVGTransform()

from gi.repository import Gtk, GLib, Gdk

class TestView(view.SvgPathWidget):
    def __init__(self, model, transform):
        super(TestView,self).__init__(model, transform)
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

    def _on_button_press_event(self, da, event):
        if self._model.num_paths > 1:
            return False

        if event.type == Gdk.EventType.BUTTON_PRESS and event.button == 1:
            seg, ratio = self._model.connect_closest(p=None,px=event.x,py=event.y)
            self._model.start_move_from_ratio(ratio)
            return True

    def get_vec_and_points(self):
        if self._model.num_paths > 1:
            return [],[],[]

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
        if self._model.num_paths > 1:
            return []

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

        try:
            self._model = model.MovingPointSvgPath(path)
        except model.MultiplePathSvgError:
            self._model = model.MultipleSvgPath(path)

        self._view = TestView(self._model, XFORM)

        self._w.add(self._view)
        self._w.show_all()

        self._step = 0.01
        GLib.timeout_add(200, self._move_along)

        try:
            self._hit = self._model.get_hitmanager(XFORM, validate=True)
            self._view.connect('motion-notify-event', self._on_motion_notify_event)
        except ImportError:
            pass
        except model.PathError as e:
            print "Could not initialize hit tester: %s" % e.message

    def _move_along(self):
        if self._model.num_paths == 1:
            self._model.advance_point(self._step, wrap=True)
        self._view.queue_draw()
        return True

    def _on_motion_notify_event(self, v, event):
        if (self._hit is not None) and self._hit.contains_px(event.x, event.y):
            print "IN"
        else:
            print "OUT"

if __name__ == "__main__":
    import os.path
    plain = os.path.join(os.path.dirname(__file__),'..','..','data','svgpaths','plain.svg')
    t = Tester(sys.argv[1] if len(sys.argv) > 1 else plain)
    Gtk.main()
