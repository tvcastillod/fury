"""Testing file unittest."""

import operator
import sys
import warnings

import numpy as np
import numpy.testing as npt

from fury import window
from fury.lib import Actor2D
import fury.testing as ft
from fury.ui.core import UI


def test_callback():
    events_name = [
        "CharEvent",
        "MouseMoveEvent",
        "KeyPressEvent",
        "KeyReleaseEvent",
        "LeftButtonPressEvent",
        "LeftButtonReleaseEvent",
        "RightButtonPressEvent",
        "RightButtonReleaseEvent",
        "MiddleButtonPressEvent",
        "MiddleButtonReleaseEvent",
    ]

    class SimplestUI(UI):
        def __init__(self):
            super(SimplestUI, self).__init__()

        def _setup(self):
            self.actor = Actor2D()

        def _set_position(self, coords):
            self.actor.SetPosition(*coords)

        def _get_size(self):
            return

        def _get_actors(self):
            return [self.actor]

        def _add_to_scene(self, _scene):
            return

    simple_ui = SimplestUI()
    current_size = (900, 600)
    scene = window.Scene()
    show_manager = window.ShowManager(
        scene=scene, size=current_size, title="FURY GridUI"
    )
    scene.add(simple_ui)
    event_counter = ft.EventCounter()
    event_counter.monitor(simple_ui)
    events_name = ["{0} 0 0 0 0 0 0 0".format(name) for name in events_name]
    events_str = "# StreamVersion 1\n" + "\n".join(events_name)
    show_manager.play_events(events_str)
    npt.assert_equal(len(event_counter.events_counts), len(events_name))


def test_captured_output():
    def foo():
        print("hello world!")

    with ft.captured_output() as (out, _):
        foo()

    npt.assert_equal(out.getvalue().strip(), "hello world!")


def test_assert():
    npt.assert_raises(
        AssertionError, ft.assert_false, True, msg="True is not false", op=operator.eq
    )
    npt.assert_raises(
        AssertionError, ft.assert_true, False, msg="False is not true", op=operator.eq
    )
    npt.assert_raises(
        AssertionError, ft.assert_less, 2, 1, msg="{0} < {1}", op=operator.lt
    )
    npt.assert_raises(
        AssertionError, ft.assert_less_equal, 2, 1, msg="{0} =< {1}", op=operator.le
    )
    npt.assert_raises(
        AssertionError, ft.assert_greater, 1, 2, msg="{0} > {1}", op=operator.gt
    )
    npt.assert_raises(
        AssertionError, ft.assert_greater_equal, 1, 2, msg="{0} >= {1}", op=operator.ge
    )
    npt.assert_raises(AssertionError, ft.assert_not_equal, 5, 5, msg="", op=operator.ne)
    npt.assert_raises(AssertionError, ft.assert_operator, 2, 1, msg="", op=operator.eq)

    arr = [np.arange(k) for k in range(2, 12, 3)]
    arr2 = [np.arange(k) for k in range(2, 12, 4)]
    npt.assert_raises(AssertionError, ft.assert_arrays_equal, arr, arr2)


def assert_warn_len_equal(mod, n_in_context):
    mod_warns = mod.__warningregistry__
    # Python 3 appears to clear any pre-existing warnings of the same type,
    # when raising warnings inside a catch_warnings block. So, there is a
    # warning generated by the tests within the context manager, but no
    # previous warnings.
    if "version" in mod_warns:
        npt.assert_equal(len(mod_warns), 2)  # including 'version'
    else:
        npt.assert_equal(len(mod_warns), n_in_context)


def test_clear_and_catch_warnings():
    # Initial state of module, no warnings
    my_mod = sys.modules[__name__]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        pass

    npt.assert_equal(getattr(my_mod, "__warningregistry__", {}), {})
    with ft.clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter("ignore")
        warnings.warn("Some warning", stacklevel=1)
    npt.assert_equal(my_mod.__warningregistry__, {})
    # Without specified modules, don't clear warnings during context
    with ft.clear_and_catch_warnings():
        warnings.warn("Some warning", stacklevel=1)
    assert_warn_len_equal(my_mod, 1)
    # Confirm that specifying module keeps old warning, does not add new
    with ft.clear_and_catch_warnings(modules=[my_mod]):
        warnings.warn("Another warning", stacklevel=1)
    assert_warn_len_equal(my_mod, 1)
    # Another warning, no module spec does add to warnings dict, except on
    # Python 3 (see comments in `assert_warn_len_equal`)
    with ft.clear_and_catch_warnings():
        warnings.warn("Another warning", stacklevel=1)
    assert_warn_len_equal(my_mod, 2)
