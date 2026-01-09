from textual import work
from textual.binding import Binding
from client.ui.components import Table
from client.core.errors import FrontError
from schemas.assignment import AssignmentHistory
from client.core.session import session
from client.api.assignment import get_assignment_history


class AssignmentHistoriesScreen(Table):
    """Screen displaying all assignment histories in a table."""
    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("f", "filter", "Filter", show=True),
    ]

    FILTER_MODES = ["all", "own"]

    def __init__(self):
        super().__init__(
            name="Workers",
            max_length=24,
            schema=AssignmentHistory
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

    # How should I do this, the best way would be to map a button on task
    # screen to navigate over here

    # TODO: implement pagination later, right now this is enough
