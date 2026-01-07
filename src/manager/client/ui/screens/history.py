from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, DataTable, Static


class AssignmentHistoriesScreen(Container):
    """Screen displaying assignment histories."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Assignment Histories", classes="screen-title"),
            VerticalScroll(
                DataTable(id="histories_table"),
                classes="content-container"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#histories_table", DataTable)
        table.add_columns("ID", "Task ID", "Worker ID",
                          "Status", "Created At", "Log")

        for history in ASSIGNMENT_HISTORIES:
            table.add_row(
                history.id,
                history.task_id,
                history.worker_id,
                history.status.value,
                history.created_at.strftime("%Y-%m-%d %H:%M"),
                history.log
            )
