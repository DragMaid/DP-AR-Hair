from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Label, Button
from textual.containers import Vertical
from textual.binding import Binding


class ErrorModal(ModalScreen):
    CSS = """
    ErrorModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.55);
    }

    .modal {
        width: 52;
        height: auto;
        padding: 2 3;
        background: $panel;
        border: round $error;
    }

    .title {
        text-align: center;
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }

    .message {
        text-align: center;
        margin-bottom: 2;
    }

    .actions {
        height: auto;
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "close", "Close"),
    ]

    def __init__(
            self,
            title: str = "Error",
            message: str = "Something went wrong"):
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label(self.title, classes="title")
            yield Label(self.message, classes="message")
            with Vertical(classes="actions"):
                yield Button("OK", id="ok", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.dismiss()

    def action_close(self) -> None:
        self.dismiss()
