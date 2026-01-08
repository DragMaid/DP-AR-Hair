from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, ListView, ListItem
from textual.message import Message
from textual.reactive import reactive
from typing import List
from textual import on


class Sidebar(Vertical):
    past_notices: reactive[List[str]] = reactive(list, recompose=True)

    NOTICE_LIMIT = 100
    SCREENS = [
        {"name": "Workers", "id": "workers"},
        {"name": "Tasks", "id": "tasks"},
        {"name": "Assignments", "id": "assignments"},
        {"name": "Histories", "id": "histories"},
        {"name": "Admins", "id": "admins"}
    ]

    class NavSelected(Message):
        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    def compose(self):
        items = []
        for i in range(len(self.SCREENS)):
            items.append(
                ListItem(
                    Static(f"{i+1} {self.SCREENS[i]['name']}"),
                    id=self.SCREENS[i]["id"]
                )
            )
        yield Static("DQS", classes="title")
        yield ListView(
            *items,
            initial_index=1,
            id="nav_list"
        )
        n = len(self.past_notices)
        self.list_view = ListView(
            *[ListItem(Static(self.past_notices[n-i-1]))
              for i in range(n)],
        )
        yield VerticalScroll(
            self.list_view,
            classes="note_container",
            can_focus=False,
            can_focus_children=False
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.post_message(self.NavSelected(event.item.id))

    @on(NavSelected)
    def handle_nav_selected(self, message: NavSelected):
        message.stop()
        for i in range(len(self.SCREENS)):
            if self.SCREENS[i]["id"] == message.action:
                self.query_one("#nav_list", ListView).index = i

    def add_note(self, text: str) -> None:
        if len(self.past_notices) >= self.NOTICE_LIMIT:
            self.past_notices.pop(0)
        self.past_notices.append(text)
        self.mutate_reactive(Sidebar.past_notices)
