from textual.containers import Vertical
from textual.widgets import Static, ListView, ListItem
from textual.message import Message


class Sidebar(Vertical):
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

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.post_message(self.NavSelected(event.item.id))
