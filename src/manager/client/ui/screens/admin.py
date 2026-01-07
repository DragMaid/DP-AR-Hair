from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, DataTable, Static


class AdminsScreen(Container):
    """Screen displaying all administrators."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Administrators", classes="screen-title"),
            VerticalScroll(
                DataTable(id="admins_table"),
                classes="content-container"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#admins_table", DataTable)
        table.add_columns("ID", "Username", "Role", "Created At")

        for admin in ADMINS:
            table.add_row(
                admin.id,
                admin.username,
                admin.role.value,
                admin.created_at.strftime("%Y-%m-%d %H:%M")
            )
