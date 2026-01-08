from textual import on
from textual import work
from textual.binding import Binding
from client.ui.modals import ConfirmModal, FormModal
from client.ui.components import Table
from client.api.worker import create_worker, delete_worker, get_workers
from client.core.errors import FrontError
from schemas.user import User
from pydantic import BaseModel


class CreateWorkerForm(BaseModel):
    email: str


class WorkersScreen(Table):
    """Screen displaying all workers in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("c", "create", "Create", show=True),
    ]

    def __init__(self):
        super().__init__(
            name="Workers",
            max_length=24,
            schema=User
        )
        self.worker_nodes = None

    # Core logic handlers
    @work(exclusive=True)
    async def handle_reload(self):
        try:
            self.table.clear()
            self.workers_nodes = await get_workers(self.fetcher)
            for worker in self.workers_nodes:
                self.table.add_row(*self.shorten(worker.values()))
        except FrontError as e:
            self.app.show_modal(
                target="error",
                message=str(e)
            )

    @work(exclusive=True)
    @on(ConfirmModal.Confirmed)
    async def handle_confirm(self, message):
        message.stop()

        # Checks if index recording started yet
        index = self.table.cursor_row
        if index < 0:
            return

        try:
            await delete_worker(
                fetcher=self.fetcher,
                worker_id=self.workers_nodes[index]["id"]
            )
            self.handle_reload()
        except FrontError as e:
            self.app.show_modal("error", message=str(e))

    @work(exclusive=True)
    @on(FormModal.Submitted)
    async def handle_form_submiited(self, message):
        message.stop()
        try:
            await create_worker(
                self.fetcher,
                message.formdata.email
            )
            self.handle_reload()
        except FrontError as e:
            self.app.show_modal(
                target="error",
                message=str(e)
            )

    # On start
    def on_mount(self) -> None:
        self.handle_reload()

    # Actions
    def action_reload(self):
        self.handle_reload()

    def action_delete(self):
        self.app.show_modal(target="confirm")

    def action_filter(self):
        pass

    def action_create(self):
        self.app.show_modal(
            target="form",
            form_name="Create worker",
            form_schema=CreateWorkerForm
        )
