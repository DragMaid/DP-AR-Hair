from datetime import datetime, timedelta
from schemas.user import User, UserRoles
from schemas.task import Task, TaskStatus
from schemas.assignment import Assignment, AssignmentHistory, AssignmentStatus


# Sample Workers
WORKERS = [
    User(
        id="worker_001",
        username="worker1@example.com",
        role=UserRoles.WORKER,
        created_at=datetime.now() - timedelta(days=30)
    ),
    User(
        id="worker_002",
        username="worker2@example.com",
        role=UserRoles.WORKER,
        created_at=datetime.now() - timedelta(days=25)
    ),
    User(
        id="worker_003",
        username="worker3@example.com",
        role=UserRoles.WORKER,
        created_at=datetime.now() - timedelta(days=20)
    ),
]

# Sample Admins
ADMINS = [
    User(
        id="admin_001",
        username="admin@example.com",
        role=UserRoles.ADMIN,
        created_at=datetime.now() - timedelta(days=90)
    ),
    User(
        id="admin_002",
        username="superadmin@example.com",
        role=UserRoles.ADMIN,
        created_at=datetime.now() - timedelta(days=100)
    ),
]

# Sample Tasks
TASKS = [
    Task(
        id="task_001",
        driving_image_id="img_drive_001",
        reference_image_id="img_ref_001",
        result_path="/results/task_001.mp4",
        retry_count=0,
        priority=1,
        status=TaskStatus.COMPLETED,
        created_at=datetime.now() - timedelta(hours=5),
        completed_at=datetime.now() - timedelta(hours=2)
    ),
    Task(
        id="task_002",
        driving_image_id="img_drive_002",
        reference_image_id="img_ref_002",
        result_path="/results/task_002.mp4",
        retry_count=1,
        priority=2,
        status=TaskStatus.PROCESSING,
        created_at=datetime.now() - timedelta(hours=3),
        completed_at=None
    ),
    Task(
        id="task_003",
        driving_image_id="img_drive_003",
        reference_image_id="img_ref_003",
        result_path="",
        retry_count=0,
        priority=1,
        status=TaskStatus.PENDING,
        created_at=datetime.now() - timedelta(minutes=30),
        completed_at=None
    ),
]

# Sample Assignments
ASSIGNMENTS = [
    Assignment(
        id="assign_001",
        task_id="task_001",
        worker_id="worker_001",
        created_at=datetime.now() - timedelta(hours=5)
    ),
    Assignment(
        id="assign_002",
        task_id="task_002",
        worker_id="worker_002",
        created_at=datetime.now() - timedelta(hours=3)
    ),
]

# Sample Assignment Histories
ASSIGNMENT_HISTORIES = [
    AssignmentHistory(
        id="hist_001",
        task_id="task_001",
        worker_id="worker_001",
        status=AssignmentStatus.FAILED,
        created_at=datetime.now() - timedelta(hours=2),
        log="Task completed successfully"
    ),
    AssignmentHistory(
        id="hist_002",
        task_id="task_002",
        worker_id="worker_002",
        status=AssignmentStatus.FAILED,
        created_at=datetime.now() - timedelta(hours=3),
        log="Task processing in progress"
    ),
    AssignmentHistory(
        id="hist_003",
        task_id="task_001",
        worker_id="worker_001",
        status=AssignmentStatus.SUCCEED,
        created_at=datetime.now() - timedelta(hours=5),
        log="Task assigned to worker"
    ),
]
