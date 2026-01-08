from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, ListView, ListItem
from textual.message import Message
from textual.reactive import reactive
from typing import List


class Sidebar(Vertical):
    past_notices: reactive[List[str]] = reactive(list, recompose=True)
    NOTICE_LIMIT = 100

    class NavSelected(Message):
        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    def compose(self):
        yield Static("DQS", classes="title")
        yield ListView(
            ListItem(Static("1  Workers"), id="workers"),
            ListItem(Static("2  Tasks"), id="tasks"),
            ListItem(Static("3  Assignments"), id="assignments"),
            ListItem(Static("4  Histories"), id="histories"),
            ListItem(Static("5  Admins"), id="admins"),
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

    def add_note(self, text: str) -> None:
        if len(self.past_notices) >= self.NOTICE_LIMIT:
            self.past_notices.pop(0)
        self.past_notices.append(text)
        self.mutate_reactive(Sidebar.past_notices)
