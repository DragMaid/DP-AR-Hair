from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, DataTable, Static
from typing import List
from pydantic import BaseModel


class Table(Container):
    """Screen displaying all tasks in a table."""
    BINDINGS = [
        Binding("C", "copy", "Copy", show=True)
    ]

    def __init__(
        self,
            name: str,
            max_length: int,
            schema: BaseModel,
            collection_name: str
    ):
        super().__init__()
        self.cname = name
        self.max_length = max_length
        self.schema = schema
        self.collection_name = collection_name

    @property
    def fetcher(self):
        return self.app.fetcher

    def compose(self) -> ComposeResult:
        self.table = DataTable(
            id=f"{self.cname}_table",
            cursor_type="row",
            fixed_rows=0,
            zebra_stripes=True
        )
        self.table.add_columns(*self.schema.__fields__.keys())
        yield Header()
        yield Container(
            Static(self.cname, classes="screen-title"),
            VerticalScroll(
                self.table,
                classes="content-container",
                can_focus=False,
            )
        )
        yield Footer()

    def shorten(self, values: List):
        res = []
        for value in values:
            value = str(value)
            if len(value) > self.max_length:
                res.append(value[:self.max_length-3] + "...")
            else:
                res.append(value)
        return res

    def format_dict(self, dictionary: dict):
        formatted = "Result\n"
        for key, value in dictionary.items():
            formatted += f"{key}: {value}\n"
        return formatted

    def action_copy(self):
        index = self.table.cursor_row
        if index < 0:
            return
        collection = getattr(self, self.collection_name)
        self.log(collection[index])
        self.app.show_modal(
            target="note",
            message=self.format_dict(collection[index])
        )
