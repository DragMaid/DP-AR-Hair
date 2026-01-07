from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, DataTable, Static


class WorkersScreen(Container):
    """Screen displaying all workers with their status."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Workers", classes="screen-title"),
            VerticalScroll(
                DataTable(id="workers_table"),
                classes="content-container"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#workers_table", DataTable)
        table.add_columns("ID", "Username", "Role", "Created At", "Status")

        for worker in WORKERS:
            # Simulate online status (in real app, this would come from API)
            status = "🟢 Online" if hash(worker.id) % 2 == 0 else "⚫ Offline"
            table.add_row(
                worker.id,
                worker.username,
                worker.role.value,
                worker.created_at.strftime("%Y-%m-%d %H:%M"),
                status
            )
