from textual import on
from textual import work
from textual.binding import Binding
from manager.client.ui.modals import ConfirmModal, FormModal
from manager.client.ui.components import Table
from manager.client.api.worker import create_worker, delete_worker, get_workers
from manager.client.core.errors import FrontError
from manager.client.core.session import session
from manager.schemas.user import User
from pydantic import BaseModel


class CreateWorkerForm(BaseModel):
    email: str


class WorkersScreen(Table):
    """Screen displaying all workers in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("f", "filter", "Filter", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("c", "create", "Create", show=True),
    ]

    FILTER_MODES = ["all", "own"]

    def __init__(self):
        super().__init__(
            name="Workers",
            max_length=24,
            schema=User
        )
        self.worker_nodes = None
        self.filter_mode_index = 0

    # Core logic handlers
    @work(exclusive=True)
    async def handle_reload(self):
        try:
            self.table.clear()

            owner_id = None
            if self.FILTER_MODES[self.filter_mode_index] == "own":
                # If return None then not logged in -> should just crash
                owner_id = session.get_user_id()

            self.workers_nodes = await get_workers(
                self.fetcher,
                owner_id=owner_id
            )

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
            password = await create_worker(
                self.fetcher,
                message.formdata.email
            )
            self.handle_reload()
            self.app.show_modal(
                target="note",
                title="PASSWORD",
                message=password
            )
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
        index = self.filter_mode_index + 1
        index = index if index < len(self.FILTER_MODES) else 0
        self.filter_mode_index = index
        self.app.add_note(f"Filter: {self.FILTER_MODES[index]}")
        self.handle_reload()

    def action_create(self):
        self.app.show_modal(
            target="form",
            form_name="Create worker",
            form_schema=CreateWorkerForm
        )
