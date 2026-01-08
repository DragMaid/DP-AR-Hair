from textual import on
from textual import work
from textual.binding import Binding
from schemas.task import Task, TaskStatus
from client.api.tasks import get_tasks, delete_task
from client.core.errors import FrontError
from client.ui.modals import ConfirmModal
from client.ui.components import Table


class TasksScreen(Table):
    """Screen displaying all tasks in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("f", "filter", "Filter", show=True),
    ]

    MODES = [t.value for t in TaskStatus]
    current_mode_index = 0

    def __init__(self):
        super().__init__(
            name="Tasks",
            max_length=24,
            schema=Task
        )
        self.tasks = None

    # Core logic handlers
    @work(exclusive=True)
    async def handle_reload(self):
        try:
            self.table.clear()
            self.tasks = await get_tasks(
                self.fetcher, status=[self.MODES[self.current_mode_index]])
            for task in self.tasks:
                self.table.add_row(*self.shorten(task.values()))
        except FrontError as e:
            self.app.show_modal("error", message=str(e))

    @work(exclusive=True)
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
            self.handle_reload()
        except FrontError as e:
            self.app.show_modal("error", message=str(e))

    # On start
    def on_mount(self) -> None:
        self.handle_reload()

    # Actions
    def action_reload(self):
        self.handle_reload()

    def action_delete(self):
        self.app.show_modal("confirm")

    def action_filter(self):
        index = self.current_mode_index + 1
        index = index if index < len(self.MODES) else 0
        self.current_mode_index = index
        self.log(self.current_mode_index)
        self.app.add_note(f"Filter: {self.MODES[index]}")
        self.handle_reload()
