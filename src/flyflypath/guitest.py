import sys

import model
import view

from gi.repository import Gtk, GLib

class Tester:
    def __init__(self, path):
        self._w = Gtk.Window()
        self._w.connect("delete-event", Gtk.main_quit)

        self._model = model.MovingPointSvgPath(path)
        self._view = view.SvgPathWidget(self._model, to_pt=True)

        self._w.add(self._view)
        self._w.show_all()

        self._step = 0.01
        self._along = 0.0
        GLib.timeout_add(500, self._move_along)

    def _move_along(self):
        self._along += self._step
        self._view.move_along(self._along)
        return self._along <= 1.0



if __name__ == "__main__":
    t = Tester(sys.argv[1] if len(sys.argv) > 1 else "plain.svg")
    Gtk.main()
