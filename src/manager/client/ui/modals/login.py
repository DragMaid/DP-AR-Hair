from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Input, Button, Label
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual import work
from textual import on
from textual.binding import Binding
from client.core.errors import FrontError
from client.api.auth import authorize


class LoginModal(ModalScreen):
    CSS = """
    LoginModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.55);
    }

    .modal {
        width: 48;
        height: auto;
        padding: 2 3;
        background: $panel;
        border: round $primary;
    }

    .modal .title {
        text-align: left;
        margin-bottom: 1;
        text-style: bold;
    }

    Input {
        margin-bottom: 1;
    }

    .actions {
        height: auto;
        margin-top: 1;
        align-horizontal: center;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel")
    ]

    class Submitted(Message):
        def __init__(self, username: str, password: str) -> None:
            self.username = username
            self.password = password
            super().__init__()

    def __init__(self, dismissible=True):
        self.dismissible = dismissible
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label("Login", classes="title")
            yield Input(placeholder="Username", id="username")
            yield Input(placeholder="Password", password=True, id="password")
            with Horizontal(classes="actions"):
                yield Button("Login", variant="success", id="login")

    def action_cancel(self) -> None:
        if self.dismissible:
            self.dismiss()

    @on(Button.Pressed, "#login")
    def submit(self) -> None:
        username = self.query_one("#username", Input).value
        password = self.query_one("#password", Input).value
        self.handle_authorization(self.Submitted(username, password))

    @work(exclusive=True)
    async def handle_authorization(self, message: Submitted) -> bool:
        try:
            await authorize(
                fetcher=self.app.fetcher,
                username=message.username,
                password=message.password,
                role='admin',
            )
            self.dismiss()
        except FrontError as e:
            self.app.show_modal(
                target="error",
                message=str(e)
            )
