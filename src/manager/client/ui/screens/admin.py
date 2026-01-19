from textual import on
from textual import work
from textual.binding import Binding
from manager.client.ui.modals import ConfirmModal, FormModal
from manager.client.ui.components import Table
from manager.client.api.admin import get_admins, create_admin, delete_admin
from manager.client.core.errors import FrontError
from manager.schemas.user import User
from pydantic import BaseModel


class CreateAdminForm(BaseModel):
    username: str


class AdminsScreen(Table):
    """Screen displaying all workers in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("f", "filter", "Filter", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("c", "create", "Create", show=True),
    ]

    def __init__(self):
        super().__init__(
            name="Admins",
            max_length=24,
            schema=User, 
            collection_name="admin_nodes"
        )
        self.admin_nodes = None
        self.filter_mode_index = 0

    # Core logic handlers
    @work(exclusive=True)
    async def handle_reload(self):
        try:
            self.table.clear()
            self.admin_nodes = await get_admins(self.fetcher)
            for admin in self.admin_nodes:
                self.table.add_row(*self.shorten(admin.values()))

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
            await delete_admin(
                fetcher=self.fetcher,
                admin_id=self.admin_nodes[index]["id"]
            )
            self.handle_reload()
        except FrontError as e:
            self.app.show_modal("error", message=str(e))

    @work(exclusive=True)
    @on(FormModal.Submitted)
    async def handle_form_submiited(self, message):
        message.stop()
        try:
            password = await create_admin(
                self.fetcher,
                message.formdata.username
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

    def action_create(self):
        self.app.show_modal(
            target="form",
            form_name="Create admin",
            form_schema=CreateAdminForm
        )
