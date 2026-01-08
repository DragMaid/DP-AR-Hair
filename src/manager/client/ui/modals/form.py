from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Input, Button, Label
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual import on
from textual.binding import Binding
from pydantic import BaseModel


class FormModal(ModalScreen):
    CSS = """
    FormModal {
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
        def __init__(self, formdata: BaseModel) -> None:
            self.formdata = formdata
            super().__init__()

    def __init__(self, form_name: str, form_schema: BaseModel):
        self.form_name = form_name
        self.form_schema = form_schema
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label(self.form_name, classes="title")

            for key in self.form_schema.__fields__.keys():
                yield Input(placeholder=key.capitalize(), id=key)

            with Horizontal(classes="actions"):
                yield Button("Submit", variant="success", id="submit")

    def action_cancel(self) -> None:
        self.dismiss()

    @on(Button.Pressed, "#submit")
    def submit(self) -> None:
        form = {}
        for key in self.form_schema.__fields__.keys():
            form[key] = self.query_one(f"#{key}", Input).value
        form = self.form_schema(**form)
        self.post_message(self.Submitted(form))
