from textual import work
from textual.binding import Binding
from manager.client.ui.components import Table
from manager.client.core.errors import FrontError
from manager.schemas.assignment import AssignmentHistory
from manager.client.core.session import session
from manager.client.api.assignment import get_assignment_history


class History:
    pass


class AssignmentHistoriesScreen(Table):
    """Screen displaying all assignment histories in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("f", "filter", "Filter", show=True),
        Binding("b", "read", "Read", show=True),
    ]

    FILTER_MODES = ["all", "own"]

    def __init__(self):
        super().__init__(
            name="Histories",
            max_length=30,
            schema=AssignmentHistory,
            collection_name="histories"
        )
        self.histories = None
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

            self.histories = await get_assignment_history(
                self.fetcher,
                owner_id=owner_id
            )

            for hist in self.histories:
                self.table.add_row(*self.shorten(hist.values()))

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

    def action_read(self):
        index = self.table.cursor_row
        if index < 0:
            return
        self.log(self.histories[index]["log"])
        self.app.show_modal(
            target="note",
            message=self.histories[index]["log"]
        )

    # TODO: implement pagination later, right now this is enough
