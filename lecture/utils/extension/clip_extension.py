# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import weakref
from abc import abstractmethod

import omni.ext
import omni.ui as ui
from omni.isaac.core import World

from omni.isaac.ui.menu import make_menu_item_description
from omni.isaac.ui.ui_utils import btn_builder, get_style, scrolling_frame_builder, setup_ui_headers
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from omni.isaac.ui.element_wrappers import (
    Button,
    CheckBox,
    CollapsableFrame,
    ColorPicker,
    DropDown,
    FloatField,
    IntField,
    StateButton,
    StringField,
    TextBlock,
    XYPlot,
)

import os

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async

class ClipExtensionState:
    init=0
    find_object=1
    pick_object=2
    place_object=3


class ClipExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._menu_items = None
        self._buttons = None
        self._ext_id = ext_id
        self._sample = None
        self._extra_frames = []
        self._user_input_text = ""
        self.start_extension(
            menu_name="",
            submenu_name="",
            name="Clip Extension",
            title="Using Clip to Pick Target Object",
            overview="This Example use Clip model to find the best matched object",
            file_path=os.path.abspath(__file__),
        )

    def start_extension(
        self,
        menu_name: str,
        submenu_name: str,
        name: str,
        title: str,
        overview: str,
        file_path: str,
        number_of_extra_frames=1,
        window_width=500,
        keep_window_open=False,
    ):
        # Set up the menu items
        menu_items = [make_menu_item_description(self._ext_id, name, lambda a=weakref.proxy(self): a._menu_callback())]
        if menu_name == "" or menu_name is None:
            self._menu_items = menu_items
        elif submenu_name == "" or submenu_name is None:
            self._menu_items = [MenuItemDescription(name=menu_name, sub_menu=menu_items)]
        else:
            self._menu_items = [
                MenuItemDescription(
                    name=menu_name, sub_menu=[MenuItemDescription(name=submenu_name, sub_menu=menu_items)]
                )
            ]
        add_menu_items(self._menu_items, "AILAB Examples")

        self._buttons = dict()
        self._build_ui(
            name=name,
            title=title,
            overview=overview,
            file_path=file_path,
            number_of_extra_frames=number_of_extra_frames,
            window_width=window_width,
            keep_window_open=keep_window_open,
        )
        
        self.state = ClipExtensionState.init
        return

    @property
    def sample(self):
        return self._sample

    def get_frame(self, index):
        if index >= len(self._extra_frames):
            raise Exception("there were {} extra frames created only".format(len(self._extra_frames)))
        return self._extra_frames[index]

    def get_world(self):
        return World.instance()

    def get_buttons(self):
        return self._buttons

    def _build_ui(
        self, name, title, overview, file_path, number_of_extra_frames, window_width, keep_window_open
    ):
        self._window = omni.ui.Window(
            name, width=window_width, height=0, visible=keep_window_open, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            self._main_stack = ui.VStack(spacing=5, height=0)
            with self._main_stack:
                setup_ui_headers(self._ext_id, file_path, title, overview=overview)
                # Create a UI frame that prints the latest UI event.
                self._create_status_report_frame()
                
                self._create_clip_frame()
                
                
        return

    def _create_status_report_frame(self):
        self._status_report_frame = CollapsableFrame("Status Report", collapsed=False)
        with self._status_report_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._status_report_field = TextBlock(
                    "Last UI Event",
                    num_lines=3,
                    tooltip="Prints the latest change to this UI",
                    include_copy_button=True,
                )

    def _create_clip_frame(self):
        self._clip_frame = CollapsableFrame("Clip", collapsed=False)
        with self._clip_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # create text block
                self._user_input_text_block = StringField(
                    "User Input Text",
                    # default_value="",
                    tooltip="Type a string",
                    read_only=False,
                    multiline_okay=False,
                    on_value_changed_fn=self._on_string_field_value_changed_fn,
                )
                
                # create button
                self._find_object_button = Button(
                    "Find Object",
                    "Find Object",
                    tooltip="Click this button to find the best matched object",
                    on_click_fn=self._on_find_object_button_clicked_fn,
                )

                self._pick_object_button = Button(
                    "Pick Object",
                    "Pick Object",
                    tooltip="Click this button to pick the best matched object",
                    on_click_fn=self._on_pick_object_button_clicked_fn,
                )
                self._pick_object_button.enabled = False



    def _set_button_tooltip(self, button_name, tool_tip):
        self._buttons[button_name].set_tooltip(tool_tip)
        return

    @abstractmethod
    def post_reset_button_event(self):
        return

    @abstractmethod
    def post_load_button_event(self):
        return

    @abstractmethod
    def post_clear_button_event(self):
        return

    def _enable_all_buttons(self, flag):
        for btn_name, btn in self._buttons.items():
            if isinstance(btn, omni.ui._ui.Button):
                btn.enabled = flag
        return

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def _on_window(self, status):
        # if status:
        return

    def on_shutdown(self):
        self._extra_frames = []
        # if self._sample._world is not None:
        #     self._sample._world_cleanup()
        if self._menu_items is not None:
            self._sample_window_cleanup()
        if self._buttons is not None:
            self._buttons["Load World"].enabled = True
            self._enable_all_buttons(False)
        self.shutdown_cleanup()
        return

    def shutdown_cleanup(self):
        return

    def _sample_window_cleanup(self):
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None
        self._menu_items = None
        self._buttons = None
        return

    def on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.CLOSED):
            if World.instance() is not None:
                self.sample._world_cleanup()
                self.sample._world.clear_instance()
                if hasattr(self, "_buttons"):
                    if self._buttons is not None:
                        self._enable_all_buttons(False)
                        self._buttons["Load World"].enabled = True
        return

    def _reset_on_stop_event(self, e):
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            self._buttons["Load World"].enabled = False
            self._buttons["Reset"].enabled = True
            self.post_clear_button_event()
        return

    def enable_pick_object_button(self, flag):
        self._pick_object_button.enabled = flag
        return

    ######################################################################################
    # Functions Below This Point Are Callback Functions Attached to UI Element Wrappers
    ######################################################################################
    def _on_string_field_value_changed_fn(self, new_value: str):
        status = f"Value was changed in string field to \n{new_value}"
        self._user_input_text = new_value
        self._status_report_field.set_text(status)

    def _on_find_object_button_clicked_fn(self):
        self._status_report_field.set_text("Find Object Button Clicked\nFind Best Matched Object\n{}".format(self._user_input_text))
        self.state = ClipExtensionState.find_object
        return
    
    def _on_pick_object_button_clicked_fn(self):
        self._status_report_field.set_text("Pick Object Button Clicked\Pick Best Matched Object\n{}".format(self._user_input_text))
        self.state = ClipExtensionState.pick_object
        return
    

