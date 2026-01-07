from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual import on


class ConfirmModal(ModalScreen):
    """A modal to confirm an action."""

    CSS = """
    ConfirmModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }

    .modal {
        width: 52;
        padding: 2 3;
        height: auto;
        background: $panel;
        border: round $primary;
    }

    .title {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    .description {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }

    .actions {
        align: center middle;
        height: auto;
    }

    .actions Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    class Confirmed(Message):
        """Message sent when the user confirms."""
        pass

    class Cancelled(Message):
        """Message sent when the user cancels."""
        pass

    def __init__(
            self,
            title: str = "Are you sure?",
            description: str = "This might lead to some stinky consequences!!"):
        super().__init__()
        self.title = title
        self.description = description

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label(self.title, classes="title")
            if self.description:
                yield Label(self.description, classes="description")
            with Horizontal(classes="actions"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Confirm", id="confirm", variant="error")

    @on(Button.Pressed, "#confirm")
    def on_confirm_pressed(self) -> None:
        self.post_message(self.Confirmed())
        self.dismiss()

    @on(Button.Pressed, "#cancel")
    def on_cancel_pressed(self) -> None:
        self.post_message(self.Cancelled())
        self.dismiss()
