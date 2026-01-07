from textual import on
from textual import work
from textual.binding import Binding
from schemas.task import Task
from client.api.tasks import get_tasks, delete_task
from client.core.errors import FrontError, ErrorCategories
from client.ui.modals import ConfirmModal
from client.ui.components import Table


class TasksScreen(Table):
    """Screen displaying all tasks in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("f", "filter", "Filter", show=True),
        Binding("ctrl+f", "find", "Find", show=True)
    ]

    def __init__(self):
        super().__init__(
            name="Tasks",
            max_length=24,
            schema=Task
        )
        self.tasks = None

    # Delete confirmation
    @work(exclusive=True)
    async def reload(self):
        try:
            self.table.clear()
            self.tasks = await get_tasks(self.fetcher)
            for task in self.tasks:
                self.table.add_row(*self.shorten(task.values()))
        except FrontError as e:
            self.app.show_modal("error", message=str(e))

    @on(ConfirmModal.Confirmed)
    async def handle_confirm(self, message):
        message.stop()

        # Checks if index recording started yet
        index = self.table.cursor_row
        if index < 0:
            return

        try:
            await delete_task(
                fetcher=self.fetcher,
                task_id=self.tasks[index]["id"]
            )
            self.reload()
        except FrontError as e:
            if e.category == ErrorCategories.CLIENT_ERROR:
                self.app.show_modal("login")
            else:
                self.app.show_modal("error", message=str(e))

    # On start
    def on_mount(self) -> None:
        self.reload()

    def on_data_table_row_highlighted(self) -> None:
        self.log(self.table.cursor_row)

    # Actions
    def action_reload(self):
        self.reload()

    def action_delete(self):
        self.app.show_modal("confirm")

    def action_filter(self):
        pass

    def action_find(self):
        pass
