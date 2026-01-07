from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal
from client.ui.components import Sidebar
from client.ui.screens import (
    WorkersScreen,
    TasksScreen,
    AssignmentsScreen,
    AssignmentHistoriesScreen,
    AdminsScreen,
)
from client.core.session import session
from client.api.fetcher import APIFetcher
from client.core.config import settings
from client.ui.modals import LoginModal, ErrorModal, ConfirmModal
from httpx import AsyncClient

from textual import on


class DQSDashboard(App):

    CSS_PATH = "client/style.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("l", "login()", "Login", show=True),
        Binding("1", "nav('workers')"),
        Binding("2", "nav('tasks')"),
        Binding("3", "nav('assignments')"),
        Binding("4", "nav('histories')"),
        Binding("5", "nav('admins')"),
    ]

    def compose(self):
        with Horizontal():
            self.sidebar = Sidebar()
            yield self.sidebar
            self.content_area = Horizontal(id="content_area")
            yield self.content_area

    def on_mount(self):
        self.client = AsyncClient()
        self.fetcher = APIFetcher(
            base_url=settings.BASE_URL,
            client=self.client,
            session=session,
            strict=False
        )
        self.current_screen = None
        self.show_screen("tasks")

    def on_nav_selected(self, message: Sidebar.NavSelected) -> None:
        self.show_screen(message.action)

    def action_nav(self, target: str) -> None:
        self.show_screen(target)

    def action_login(self) -> None:
        self.show_modal("login")

    # Utility functions
    def show_modal(self, target: str, **kwargs) -> None:
        modal_map = {
            "error": ErrorModal,
            "login": LoginModal,
            "confirm": ConfirmModal
        }
        modal_cls = modal_map.get(target)
        if modal_cls:
            self.push_screen(modal_cls(**kwargs))

    def show_screen(self, target: str) -> None:
        screen_map = {
            "workers": WorkersScreen,
            "tasks": TasksScreen,
            "assignments": AssignmentsScreen,
            "histories": AssignmentHistoriesScreen,
            "admins": AdminsScreen,
        }

        screen_cls = screen_map.get(target)
        if screen_cls:
            if self.current_screen:
                self.current_screen.remove()
            self.current_screen = screen_cls()
            self.content_area.mount(self.current_screen)

    # Re-routing confirm message to active screen
    @on(ConfirmModal.Confirmed)
    def handle_confirm(self, message):
        if self.current_screen:
            self.current_screen.post_message(message)


if __name__ == "__main__":
    DQSDashboard().run()
