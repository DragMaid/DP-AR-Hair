from textual import on
from textual import work
from textual.binding import Binding
from manager.schemas.task import Task, TaskStatus
from manager.client.api.tasks import get_tasks, delete_task
from manager.client.core.errors import FrontError
from manager.client.ui.modals import ConfirmModal
from manager.client.ui.components import Table


class TasksScreen(Table):
    """Screen displaying all tasks in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("f", "filter", "Filter", show=True),
    ]

    FILTER_MODES = ["all"] + [t.value for t in TaskStatus]

    def __init__(self):
        super().__init__(
            name="Tasks",
            max_length=24,
            schema=Task
        )
        self.tasks = None
        self.filter_mode_index = 0

    # Core logic handlers
    @work(exclusive=True)
    async def handle_reload(self):
        try:
            mode = self.FILTER_MODES[self.filter_mode_index]
            self.table.clear()
            self.tasks = await get_tasks(
                self.fetcher,
                status=None if mode == "all" else [mode]
            )
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
        index = self.filter_mode_index + 1
        index = index if index < len(self.FILTER_MODES) else 0
        self.filter_mode_index = index
        self.app.add_note(f"Filter: {self.FILTER_MODES[index]}")
        self.handle_reload()
