# ailabktw
# 

from abc import abstractmethod
import omni.ext
import omni.ui as ui

from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
import weakref

from omni.isaac.ui.ui_utils import setup_ui_headers, get_style, btn_builder, scrolling_frame_builder
import asyncio

from omni.isaac.examples.ailab_script import AILab
from omni.isaac.core import World

import os

# ----


#
class AILabExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self._ext_id= 'omni.isaac.examples-1.5.1'
        self.use_custom_updated = False

        self.current_target = None
        pass

    def on_startup(self, ext_id: str='omni.isaac.examples-1.5.1'):
        self._menu_items = None
        self._buttons = None
        self._ext_id = ext_id
        self._sample = None
        self._extra_frames = []
        return
    
    def on_custom_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="AILab extension",
            title="AILab extension Example",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_core_hello_world.html",
            overview="This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            file_path=os.path.abspath(__file__),
            sample=AILab(),
        )
        return

    def start_extension(
        self,
        menu_name: str,
        submenu_name: str,
        name: str,
        title: str,
        doc_link: str,
        overview: str,
        file_path: str,
        sample=None,
        number_of_extra_frames=1,
        window_width=420,
        keep_window_open=False,
    ):
        if sample is None:
            self._sample = AILab()
        else:
            self._sample = sample

        menu_items = [MenuItemDescription(name=name, onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())]
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
        add_menu_items(self._menu_items, "Isaac Examples")

        self._buttons = dict()
        self._build_ui(
            name=name,
            title=title,
            doc_link=doc_link,
            overview=overview,
            file_path=file_path,
            number_of_extra_frames=number_of_extra_frames,
            window_width=window_width,
            keep_window_open=keep_window_open,
        )
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
        self, name, title, doc_link, overview, file_path, # number_of_extra_frames, window_width, keep_window_open,
        number_of_extra_frames=1,
        window_width=420,
        keep_window_open=False,
        #
        use_custom_update=False,
        **args
    ):
        self._window = omni.ui.Window(
            name, width=window_width, height=0, visible=keep_window_open, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                setup_ui_headers(self._ext_id, file_path, title, doc_link, overview)
                self._controls_frame = ui.CollapsableFrame(
                    title="Object Picking",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    for i in range(number_of_extra_frames):
                        self._extra_frames.append(
                            ui.CollapsableFrame(
                                title="",
                                width=ui.Fraction(0.33),
                                height=0,
                                visible=False,
                                collapsed=False,
                                style=get_style(),
                                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                            )
                        )
                with self._controls_frame:
                    # = ++
                    # if use_custom_update:
                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        dict = {
                            "label": "Check Objects",
                            "type": "button",
                            "text": "Check",
                            "tooltip": "Check object to pick",
                            "on_clicked_fn": self._on_check,
                        }
                        self._buttons["Check Objects"] = btn_builder(**dict)
                        self._buttons["Check Objects"].enabled = True

                        dict = {
                            "label": "Reset",
                            "type": "button",
                            "text": "Reset",
                            "tooltip": "Reset robot and environment",
                            "on_clicked_fn": self._on_reset,
                        }
                        self._buttons["Reset"] = btn_builder(**dict)
                        self._buttons["Reset"].enabled = False
                        
                        dict = {
                            "label": "Object No.0",
                            "type": "button",
                            "text": "000_unkown",
                            "tooltip": "Pick object and move",
                            "on_clicked_fn": self._on_button_00,
                        }
                        self._buttons["Object No.0"] = btn_builder(**dict)
                        self._buttons["Object No.0"].enabled = False

                        dict = {
                            "label": "Object No.1",
                            "type": "button",
                            "text": "001_unkown",
                            "tooltip": "Pick object and move",
                            "on_clicked_fn": self._on_button_01,
                        }
                        self._buttons["Object No.1"] = btn_builder(**dict)
                        self._buttons["Object No.1"].enabled = False

                        dict = {
                            "label": "Object No.2",
                            "type": "button",
                            "text": "002_unkown",
                            "tooltip": "Pick object and move",
                            "on_clicked_fn": self._on_button_02,
                        }
                        self._buttons["Object No.2"] = btn_builder(**dict)
                        self._buttons["Object No.2"].enabled = False
                    # # else:   
                    #     with ui.VStack(style=get_style(), spacing=5, height=0):
                    #         dict = {
                    #             "label": "Check Objects",
                    #             "type": "button",
                    #             "text": "Check",
                    #             "tooltip": "Check object to pick",
                    #             "on_clicked_fn": self._on_check,
                    #         }
                    #         self._buttons["Check Objects"] = btn_builder(**dict)
                    #         self._buttons["Check Objects"].enabled = True

                    #         # dict = {
                    #         #     "label": "Pick Object",
                    #         #     "type": "button",
                    #         #     "text": "Pick",
                    #         #     "tooltip": "Pick object and move",
                    #         #     "on_clicked_fn": self._on_load_world,
                    #         # }
                    #         # self._buttons["Pick Object"] = btn_builder(**dict)
                    #         # self._buttons["Pick Object"].enabled = False

                    #         dict = {
                    #             "label": "Reset",
                    #             "type": "button",
                    #             "text": "Reset",
                    #             "tooltip": "Reset robot and environment",
                    #             "on_clicked_fn": self._on_reset,
                    #         }
                    #         self._buttons["Reset"] = btn_builder(**dict)
                    #         self._buttons["Reset"].enabled = False
                            
                    #         dict = {
                    #             "label": "Object No.0",
                    #             "type": "button",
                    #             "text": "000_unkown",
                    #             "tooltip": "Pick object and move",
                    #             "on_clicked_fn": self._on_selected_button,
                    #         }
                    #         self._buttons["Object No.0"] = btn_builder(**dict)
                    #         self._buttons["Object No.0"].enabled = False

                    #         dict = {
                    #             "label": "Object No.1",
                    #             "type": "button",
                    #             "text": "001_unkown",
                    #             "tooltip": "Pick object and move",
                    #             "on_clicked_fn": self._on_selected_button,
                    #         }
                    #         self._buttons["Object No.1"] = btn_builder(**dict)
                    #         self._buttons["Object No.1"].enabled = False

                    #         dict = {
                    #             "label": "Object No.2",
                    #             "type": "button",
                    #             "text": "002_unkown",
                    #             "tooltip": "Pick object and move",
                    #             "on_clicked_fn": self._on_selected_button,
                    #         }
                    #         self._buttons["Object No.2"] = btn_builder(**dict)
                    #         self._buttons["Object No.2"].enabled = False
        return

    def _set_button_tooltip(self, button_name, tool_tip):
        self._buttons[button_name].set_tooltip(tool_tip)
        return

    def _on_load_world(self):
        async def _on_load_world_async():
            await self._sample.load_world_async()
            await omni.kit.app.get_app().next_update_async()
            self._sample._world.add_stage_callback("stage_event_1", self.on_stage_event)
            self._enable_all_buttons(True)
            self._buttons["Load World"].enabled = False
            self.post_load_button_event()
            self._sample._world.add_timeline_callback("stop_reset_event", self._reset_on_stop_event)

        asyncio.ensure_future(_on_load_world_async())
        return

    def _on_reset(self):
        async def _on_reset_async():
            await self._sample.reset_async()
            await omni.kit.app.get_app().next_update_async()
            self.post_reset_button_event()

        asyncio.ensure_future(_on_reset_async())
        return

    # ++
    def _on_check(self):
        async def _on_check():
            # await self._sample.check_async()
            # await omni.kit.app.get_app().next_update_async()
            
            self._enable_all_buttons(True)
            self._buttons["Check Objects"].enabled = False
            self.use_custom_updated = False

            # self.post_reset_button_event()

        asyncio.ensure_future(_on_check())
        return
    
    def _on_selected_button(self):
        async def _on_selected_button():
            # await self._sample.selected_button_async()
            # await omni.kit.app.get_app().next_update_async()

            self._enable_all_buttons(False)
            self._buttons["Check Objects"].enabled = True
            self.use_custom_updated = True
            
            self.post_reset_button_event()

        asyncio.ensure_future(_on_selected_button())
        return
    
    def _on_button_00(self):
        async def _on_button_00():
            # await self._sample.check_async()
            # await omni.kit.app.get_app().next_update_async()
            
            self._enable_all_buttons(False)
            self._buttons["Check Objects"].enabled = True
            self.use_custom_updated = True
            self.current_target = "task_object_name_0"
            
            self.post_reset_button_event()

        asyncio.ensure_future(_on_button_00())
        return

    def _on_button_01(self):
        async def _on_button_01():
            # await self._sample.check_async()
            # await omni.kit.app.get_app().next_update_async()
            
            self._enable_all_buttons(False)
            self._buttons["Check Objects"].enabled = True
            self.use_custom_updated = True
            self.current_target = "task_object_name_1"
            
            self.post_reset_button_event()

        asyncio.ensure_future(_on_button_01())
        return

    def _on_button_02(self):
        async def _on_button_02():
            # await self._sample.check_async()
            # await omni.kit.app.get_app().next_update_async()
            
            self._enable_all_buttons(False)
            self._buttons["Check Objects"].enabled = True
            self.use_custom_updated = True
            self.current_target = "task_object_name_2"
            
            self.post_reset_button_event()

        asyncio.ensure_future(_on_button_02())
        return


    # def _on_check(self):
    #     async def _on_check():
    #         await self._sample.check_async()
    #         await omni.kit.app.get_app().next_update_async()
    #         self.post_reset_button_event()

    #     asyncio.ensure_future(_on_check())
    #     return


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
        if self._sample._world is not None:
            self._sample._world_cleanup()
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
