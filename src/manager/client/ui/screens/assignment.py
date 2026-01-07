from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, DataTable, Static


class AssignmentsScreen(Container):
    """Screen displaying all assignments."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Assignments", classes="screen-title"),
            VerticalScroll(
                DataTable(id="assignments_table"),
                classes="content-container"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#assignments_table", DataTable)
        table.add_columns("ID", "Task ID", "Worker ID", "Created At")

        for assignment in ASSIGNMENTS:
            table.add_row(
                assignment.id,
                assignment.task_id,
                assignment.worker_id,
                assignment.created_at.strftime("%Y-%m-%d %H:%M")
            )
