# PYTHONFILE
# home.py
# AX/app/routes/home.py
# -*- coding: utf-8 -*-
import os
import importlib
import inspect
import re
import time
import logging
import asyncio
import json
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Callable, Coroutine
from pathlib import Path
from enum import Enum

# --- Core Dependencies ---
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import PlainTextResponse
import aiofiles
from pydantic import BaseModel, Field, ConfigDict

# --- Configuration ---
LOG_FORMAT = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__) # Define logger early

# --- Custom LLM Client Import ---
# Attempt to import the custom GeminiClient based on the assumed structure
GeminiClient = None
try:
    # Assumes this script is in app/routes/home.py, so app.llm.gemini works
    from app.llm.gemini import GeminiClient
    logger.info("Successfully imported custom GeminiClient from app.llm.gemini.")
except ImportError:
    logger.warning("Could not import custom GeminiClient from 'app.llm.gemini'. LLM functionality will be unavailable. Ensure the file exists and the project structure is correct.")
except Exception as e:
    logger.error(f"An unexpected error occurred during custom GeminiClient import: {e}", exc_info=True)


# --- API Key Handling ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY and GeminiClient:
    logger.warning("GEMINI_API_KEY environment variable not found. Custom GeminiClient initialization might fail if it relies on this key directly (depends on its implementation). Ensure it's set in your .env file or environment.")
elif not GeminiClient:
    logger.warning("Proceeding without GeminiClient. LLM calls will fail if attempted.")


# Type alias for the Gemini Client
# Use the imported custom client directly
GeminiModelClient = GeminiClient # Use the potentially imported custom client class

router = APIRouter()


# --- File paths ---
try:
    # Assuming this script is in app/routes/home.py, BASE_DIR should be the project root
    # (app/routes/home.py -> app/routes -> app -> project_root)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError: # Handle case where __file__ might not be defined (e.g., interactive session)
     BASE_DIR = Path(".").resolve()
     logger.warning(f"Could not reliably determine project root using __file__. Using current directory: {BASE_DIR}. Adjust if needed.")
except IndexError: # Fallback if structure is different
    BASE_DIR = Path(".").resolve()
    logger.warning(f"Could not reliably determine project root relative to routes. Using current directory: {BASE_DIR}. Adjust if needed.")

# Shared cache/log directory in project root
CACHE_DIR = BASE_DIR / ".cache"
# Subdirectory for orchestration outputs, each run gets its own folder
OUTPUT_BASE_DIR = CACHE_DIR / "output"

# Shared log files (will be referenced in metadata)
CACHE_FILE = CACHE_DIR / "cached_home.md" # Cache for the main endpoint result
TASK_LOG_FILE = CACHE_DIR / "tasks.json" # Shared across all orchestrations
STEP_LOG_FILE = CACHE_DIR / "steps.json" # Shared across all orchestrations
MEMORY_LOG_FILE = CACHE_DIR / "memory.json" # Shared memory log (Note: Current implementation keeps memory in RAM)

CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure shared cache dir exists
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True) # Ensure base output dir exists
logger.info(f"Using Project BASE_DIR: {BASE_DIR}")
logger.info(f"Shared Cache/Log location: {CACHE_DIR}")
logger.info(f"Orchestration Output Base Directory: {OUTPUT_BASE_DIR}")


CACHE_EXPIRATION = 24 * 60 * 60 # 24 hours in seconds
MAX_ITERATIONS = 10 # Max refinement iterations per step
EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4) # Thread pool for sync operations

# --- Pydantic Schemas ---

class TaskStatus(str, Enum):
    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    DONE = "Done"
    ERROR = "Error"
    CANCELLED = "Cancelled"

class TaskSchema(BaseModel):
    model_config = ConfigDict(use_enum_values=True) # Store enum values as strings in JSON
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    orchestration_id: Optional[str] = Field(None, description="Identifier for the orchestration run this task belongs to.") # Link tasks to a specific manager run
    name: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None # Name of the agent assigned
    details: Optional[str] = None # Description or context for the task
    result: Optional[Any] = None # Store results (e.g., generated markdown, file list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MemorySchema(BaseModel):
    """Represents the short-term memory of an agent. Stored IN-MEMORY only."""
    agent_name: str
    agent_role: str
    current_task_id: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list) # Log of agent actions/thoughts
    scratchpad: Optional[str] = None # Temporary working space for the agent
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_history(self, event_type: str, content: Any):
        """Adds an event to the agent's history, trimming old entries."""
        max_history = 20 # Limit history size to prevent excessive memory usage
        self.history.append({
            "type": event_type,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat() # Store as ISO string
        })
        # Trim history if it exceeds max size
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        self.last_updated = datetime.now(timezone.utc)

class ToolParameter(BaseModel):
    """Defines a parameter for a tool."""
    name: str
    description: str
    type: str # e.g., 'string', 'integer', 'boolean', 'any' (consider more specific types if needed)
    required: bool

class ToolSchema(BaseModel):
    """Defines the schema and callable reference for a tool available to agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True) # Allow Callable types
    name: str
    description: str
    parameters: List[ToolParameter]
    callable_ref: Callable[..., Any | Coroutine[Any, Any, Any]] # Reference to the function/coroutine

# --- StepSchema and StepStatus ---
class StepStatus(str, Enum):
    """Enumeration for the status of an orchestration step."""
    START = "Start"
    IN_PROGRESS = "InProgress" # If steps have sub-progress tracking
    END = "End"
    COMPLETE = "Complete" # Alias or alternative for END, often used for successful task steps
    ERROR = "Error"
    CANCELLED = "Cancelled"
    WARN = "Warn"
    INFO = "Info"
    POLLING = "Polling"
    ASSIGN = "Assign" # Status for when a task is assigned
    BLOCK = "Block" # Status when orchestration is blocked (e.g., missing dependency)
    SKIP = "Skip" # Status when a step is skipped
    SUCCESS = "Success" # Explicit success status, especially for the final orchestration step

    # Helper to convert common log level names to StepStatus
    @classmethod
    def from_log_level(cls, level_name: str):
        level_upper = level_name.strip().upper()
        if level_upper == "INFO": return cls.INFO
        if level_upper == "WARNING": return cls.WARN
        if level_upper == "ERROR": return cls.ERROR
        if level_upper == "CRITICAL": return cls.ERROR # Map critical to Error status
        # Add other mappings if needed
        return cls.INFO # Default fallback

class StepSchema(BaseModel):
    """Pydantic model for storing individual orchestration step logs."""
    model_config = ConfigDict(use_enum_values=True) # Ensure enum values are used in serialization
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    orchestration_id: str # Required: Steps always belong to an orchestration run
    task_id: Optional[str] = Field(None, description="Optional Task ID this step relates to, if applicable.") # Link step to a specific task
    agent_name: str # Name of the agent performing the step (or 'Manager Orchestrator')
    step_name: str # A descriptive name for the step (e.g., "Analyze Files", "LLM Call - Refine Analysis", "Use Tool - Create Task")
    status: StepStatus = StepStatus.INFO # Default status, indicating an informational log entry
    details: Optional[str] = Field(None, description="Detailed message or context for the step.") # More detailed message
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # Timestamp of the step log entry
    duration_ms: Optional[int] = Field(None, description="Optional duration of the step in milliseconds.") # Duration if applicable


# --- Orchestration Output Schema (Revised based on user request) ---
class AgentItem(BaseModel):
    """Represents an agent used in the orchestration process."""
    agent_name: str = Field(..., description="The name of the agent.")
    role: str = Field(..., description="The role of the agent in the orchestration process.", examples=["planner", "executor", "validator", "monitor", "Crew"])
    goal: Optional[str] = Field(None, description="The specific goal or purpose assigned to this agent within the orchestration.")
    backstory: Optional[str] = Field(None, description="A brief narrative or background context for the agent's role and capabilities.")
    tools: List[str] = Field(default_factory=list, description="A list of tool names available to the agent.", examples=[["API_Client", "DataParser", "CSVExtractor"], ["create_task", "read_task"]])

class StatisticsSchema(BaseModel):
    """Represents orchestration statistics."""
    total_iterations: Optional[int] = Field(None, description="The total number of iterations performed in the orchestration (if tracked).")
    total_steps: Optional[int] = Field(None, description="The total number of steps executed during the orchestration.")
    total_tasks: Optional[int] = Field(None, description="The total number of tasks created/managed in the orchestration.")
    successful_tasks: Optional[int] = Field(None, description="The number of tasks that completed successfully (status DONE).")
    failed_tasks: Optional[int] = Field(None, description="The number of tasks that failed (status ERROR or CANCELLED).")
    retried_tasks: Optional[int] = Field(None, description="The number of tasks that were retried after failure (if tracked).")
    max_parallel_tasks: Optional[int] = Field(None, description="The maximum number of tasks executed in parallel (if tracked).")
    avg_task_duration: Optional[float] = Field(None, description="The average execution time per completed task in seconds (if calculated).")
    longest_task_duration: Optional[float] = Field(None, description="The duration of the longest completed task in seconds (if calculated).")
    shortest_task_duration: Optional[float] = Field(None, description="The duration of the shortest completed task in seconds (if calculated).")
    total_execution_time: Optional[float] = Field(None, description="The total time taken for the orchestration process in seconds.")

class OrchestrationOutputSchema(BaseModel):
    """Defines the output structure and metadata for an orchestration run."""
    dir: str = Field(..., description="The relative path to the main folder containing results specific to this orchestration run.", example=".cache/output/<orchestration_id>")
    orchestration_name: str = Field(..., description="The name of the orchestration process.", example="Automated Documentation Generation")
    orchestration_goal: str = Field(..., description="The main goal of this orchestration run.", example="Generate project documentation based on code analysis.")
    orchestration_instruction: str = Field(..., description="A high-level description or set of instructions defining the orchestration workflow.", example="Analyze files -> Generate analysis -> Create structure -> Enhance design -> Add KB -> Polish -> Save result.")
    step_log: str = Field(..., description="The relative path to the shared JSON file containing logs for all orchestration steps (filter by 'orchestration_id' within the file).", example=".cache/steps.json")
    task_log: str = Field(..., description="The relative path to the shared JSON file containing logs for all tasks (filter by 'orchestration_id' within the file).", example=".cache/tasks.json")
    memory_log: Optional[str] = Field(None, description="The relative path to the shared JSON file potentially storing memory snapshots (filter by 'orchestration_id' if implemented). Currently N/A.", example=".cache/memory.json")
    agent_list: List[AgentItem] = Field(..., description="A list of agents used in this orchestration process, including their roles, goals, and backstories.")
    statistics: StatisticsSchema = Field(..., description="Runtime statistics specific to this orchestration run.")
    final_result_file_type: List[str] = Field(..., description="The file format(s) of the primary result file(s) produced by the orchestration.", examples=[["md"], ["txt"], ["json"]])
    final_result: str = Field(..., description="The name of the primary final result file (e.g., 'result.md') relative to the orchestration's output directory ('dir').", example="result.md")


# --- StepManager Class ---
class StepManager:
    """
    Manages the logging of orchestration steps to a persistent JSON file (`steps.json`).
    Provides methods to initialize (load existing logs), add new step logs, and save changes.
    Logs are shared across all orchestration runs.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        # Use a list to store steps loaded/added during THIS process lifetime.
        # This avoids loading potentially huge shared logs entirely into memory.
        # We append directly to the file for new logs.
        self._steps_this_session: List[StepSchema] = []
        self._file_lock = asyncio.Lock() # Async lock for file append/read operations
        self._init_lock = asyncio.Lock() # Lock specifically for initialization
        self._initialized = False # Flag to track if initial checks/setup is done
        logger.info(f"[StepManager] Initialized. Shared step log file path: {self.file_path}")

    async def initialize(self):
        """
        Ensures the log file directory exists and marks the manager as initialized.
        Does NOT load all previous logs into memory.
        """
        async with self._init_lock:
            if self._initialized:
                return # Already initialized

            logger.info(f"[StepManager] Initializing shared step log: {self.file_path}")
            try:
                # Ensure the directory for the shared log file exists
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                # Optional: Check if file exists, create if not? Or just let append handle it.
                # Let's assume append will create if needed.
                logger.info(f"[StepManager] Ensured log directory exists: {self.file_path.parent}")
                self._initialized = True # Mark as initialized successfully
            except Exception as e:
                logger.error(f"[StepManager] Error during initialization (ensuring directory): {e}", exc_info=True)
                # Proceed even if directory check failed, append might still work or fail later.
                self._initialized = True # Mark initialized to prevent retry loops.

    async def create_step_log(self, step: StepSchema):
        """
        Appends a new step log entry to the shared JSON file.
        This method is designed for appending, making it more scalable for shared logs.
        """
        if not self._initialized:
            logger.warning("[StepManager] create_step_log called before initialization. Initializing now.")
            await self.initialize()

        step_dict = step.model_dump(mode='json') # Convert to dict for JSON
        step_json_line = json.dumps(step_dict, default=str) + '\n' # Convert to JSON line

        async with self._file_lock:
            try:
                async with aiofiles.open(self.file_path, "a", encoding="utf-8") as f:
                    await f.write(step_json_line)
                # Optionally keep track of steps added *this session* if needed elsewhere
                # self._steps_this_session.append(step)
                status_value = step.status.value if isinstance(step.status, Enum) else str(step.status)
                logger.debug(f"[StepManager] Appended step log: OrchID={step.orchestration_id}, Step={step.step_name}, Status={status_value}")
            except Exception as e:
                logger.error(f"[StepManager] Error appending step log to {self.file_path}: {e}", exc_info=True)

    async def get_steps(self, orchestration_id: Optional[str] = None, limit: int = 1000) -> List[StepSchema]:
        """
        Retrieves step logs from the shared file, optionally filtered and limited.
        Reads the file line by line for better memory efficiency with large logs.
        NOTE: This reads the *entire* log file if no orchestration_id is provided,
              which can be slow for very large shared logs. Consider dedicated storage
              or indexing for performance-critical scenarios.
        """
        if not self._initialized:
            await self.initialize()

        matched_steps = []
        async with self._file_lock: # Lock during read to avoid conflicts with writes
            if not os.path.exists(self.file_path):
                 logger.warning(f"[StepManager] Step log file {self.file_path} not found during get_steps.")
                 return []
            try:
                async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
                    async for line in f:
                        if not line.strip(): continue # Skip empty lines
                        try:
                            step_dict = json.loads(line)
                            # Filter by orchestration ID if provided
                            if orchestration_id and step_dict.get('orchestration_id') != orchestration_id:
                                continue

                            # Convert timestamp string back to datetime
                            if 'timestamp' in step_dict and isinstance(step_dict['timestamp'], str):
                                ts_str = step_dict['timestamp'].replace("Z", "+00:00")
                                step_dict['timestamp'] = datetime.fromisoformat(ts_str)

                            # Validate and create StepSchema
                            step_obj = StepSchema(**step_dict)
                            matched_steps.append(step_obj)

                        except (json.JSONDecodeError, ValueError, TypeError) as parse_err:
                            logger.warning(f"[StepManager] Skipping invalid step log line during read: {parse_err}. Line: '{line[:100]}...'")
                        # Stop reading if limit is reached (when filtering)
                        # Note: If no filter, we read whole file then limit. If filtering, we limit as we go.
                        if orchestration_id and len(matched_steps) >= limit:
                             logger.debug(f"[StepManager] Reached read limit ({limit}) while filtering for orchestration_id {orchestration_id}")
                             break

            except Exception as e:
                 logger.error(f"[StepManager] Error reading steps from {self.file_path}: {e}", exc_info=True)
                 return [] # Return empty list on error

        # Sort results *after* reading and filtering
        matched_steps.sort(key=lambda s: s.timestamp, reverse=True)

        # Apply limit if not already applied during filtering
        if not orchestration_id:
             final_steps = matched_steps[:limit]
        else:
             final_steps = matched_steps # Already limited during read if filtering

        logger.debug(f"[StepManager] Retrieved {len(final_steps)} steps (Filter: {orchestration_id or 'None'}, Limit: {limit})")
        return final_steps


# --- Task Manager Class (as Tool Provider) ---
class TaskManager:
    """
    Manages the lifecycle of tasks (CRUD operations) and provides these operations
    as tools for agents. Tasks are persisted to a shared JSON file.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._tasks: Dict[str, TaskSchema] = {} # In-memory storage (cache) of tasks known to this instance
        self._lock = asyncio.Lock() # Async lock for safe concurrent access to memory cache and file
        self._initialized = False # Flag to track if tasks have been loaded from file initially
        logger.info(f"[TaskManager] Initialized. Shared task file path: {self.file_path}")

    async def initialize(self):
        """Loads tasks from the JSON file into memory. Should be called before first use."""
        async with self._lock:
            if self._initialized:
                return # Already initialized
            logger.info(f"[TaskManager] Initializing and loading tasks from shared file: {self.file_path}")
            try:
                # Ensure directory exists
                self.file_path.parent.mkdir(parents=True, exist_ok=True)

                if os.path.exists(self.file_path):
                    async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        if content:
                            loaded_tasks_list = json.loads(content)
                            valid_tasks = {}
                            for task_dict in loaded_tasks_list:
                                try:
                                    # Convert ISO string timestamps back to datetime objects
                                    if 'created_at' in task_dict and isinstance(task_dict['created_at'], str):
                                        ts_str = task_dict['created_at'].replace("Z", "+00:00")
                                        task_dict['created_at'] = datetime.fromisoformat(ts_str)
                                    if 'updated_at' in task_dict and isinstance(task_dict['updated_at'], str):
                                        ts_str = task_dict['updated_at'].replace("Z", "+00:00")
                                        task_dict['updated_at'] = datetime.fromisoformat(ts_str)

                                    # Validate and create TaskSchema instance
                                    task_obj = TaskSchema(**task_dict)
                                    valid_tasks[task_obj.id] = task_obj
                                except (ValueError, TypeError) as val_err: # Catch Pydantic validation errors or date parsing errors
                                    logger.warning(f"[TaskManager] Skipping invalid task data during load: ID={task_dict.get('id', 'N/A')}, Error: {val_err}")
                            self._tasks = valid_tasks # Populate in-memory cache
                            logger.info(f"[TaskManager] Loaded {len(self._tasks)} valid tasks from {self.file_path} into memory cache.")
                        else:
                            logger.info(f"[TaskManager] Task file {self.file_path} is empty. Starting with no tasks in cache.")
                            self._tasks = {}
                else:
                    logger.info(f"[TaskManager] Task file {self.file_path} not found. Starting with no tasks in cache.")
                    self._tasks = {}
                self._initialized = True
            except (json.JSONDecodeError, IOError, Exception) as e:
                logger.error(f"[TaskManager] Error loading tasks from {self.file_path}: {e}. Starting with no tasks in cache.", exc_info=True)
                self._tasks = {} # Reset cache on error
                self._initialized = True # Mark as initialized even on error to prevent reload loops

    async def _save_tasks(self):
        """Saves the current *entire* in-memory cache of tasks back to the shared JSON file."""
        # Note: This overwrites the file. For high concurrency, append-based or DB storage is better.
        if not self._initialized:
            logger.warning("[TaskManager] Attempted to save tasks before initialization.")
            return
        async with self._lock: # Ensure save operation is atomic for this instance
            logger.debug(f"[TaskManager] Saving {len(self._tasks)} tasks from memory cache to shared file {self.file_path}")
            try:
                # Ensure the cache directory exists before attempting to write
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert tasks to dicts for JSON serialization, ensuring datetime is handled
                tasks_list = [task.model_dump(mode='json') for task in self._tasks.values()]
                async with aiofiles.open(self.file_path, "w", encoding="utf-8") as f:
                    # Use default=str as a fallback for types json doesn't handle directly
                    await f.write(json.dumps(tasks_list, indent=2, default=str))
                logger.debug(f"[TaskManager] Saved {len(tasks_list)} tasks to {self.file_path}")
            except (IOError, TypeError, Exception) as e:
                logger.error(f"[TaskManager] Error saving tasks to {self.file_path}: {e}", exc_info=True)


    # --- CRUD Methods (Internal implementations used by Tools) ---
    async def _create_task_impl(
        self,
        name: str,
        details: str = "",
        assigned_to: Optional[str] = None,
        orchestration_id: Optional[str] = None # Added parameter for linking
    ) -> TaskSchema:
        """Internal method to create a new task instance."""
        if not self._initialized: await self.initialize() # Ensure loaded
        async with self._lock:
            new_task = TaskSchema(
                name=name,
                details=details,
                assigned_to=assigned_to,
                orchestration_id=orchestration_id # Pass orchestration ID to the task
            )
            self._tasks[new_task.id] = new_task # Add to in-memory cache
            log_msg = f"[TaskManager] Created Task ID: {new_task.id}"
            if orchestration_id:
                 log_msg += f", OrchID: {orchestration_id}" # Update log message
            log_msg += f", Name: '{new_task.name}', Status: {new_task.status.value}"
            logger.info(log_msg)
            await self._save_tasks() # Persist changes to shared file
            return new_task.model_copy(deep=True) # Return a copy to prevent external modification

    async def _read_task_impl(self, task_id: str) -> Optional[TaskSchema]:
        """Internal method to read a single task by its ID from the in-memory cache."""
        if not self._initialized: await self.initialize()
        # Reading from cache is generally safe without lock if mutation uses lock, but lock ensures consistency during init
        async with self._lock: # Use lock mainly to ensure cache is loaded before read
            task = self._tasks.get(task_id)
            # Optionally: Could try reloading from file if not in cache? - Adds complexity.
            # For now, rely on the cache reflecting the state known to this instance.
            return task.model_copy(deep=True) if task else None # Return a copy

    async def _read_all_tasks_impl(
        self,
        status_filter: Optional[TaskStatus] = None,
        assigned_filter: Optional[str] = None,
        orchestration_id_filter: Optional[str] = None # Optional filter by orchestration ID
        ) -> List[TaskSchema]:
        """Internal method to read multiple tasks from the in-memory cache, with optional filters."""
        if not self._initialized: await self.initialize()
        async with self._lock: # Lock to ensure cache consistency during read/filter
             # Create a snapshot of tasks under lock
             tasks_snapshot = list(self._tasks.values())

        # Apply filters outside the lock on the snapshot
        filtered_tasks = tasks_snapshot
        if status_filter:
            # Ensure filter value is the correct Enum type if passed as string
            if isinstance(status_filter, str):
                try:
                    status_filter = TaskStatus(status_filter)
                except ValueError:
                     logger.warning(f"[TaskManager] Invalid status filter value received: '{status_filter}'. Returning empty list.")
                     return [] # Return empty list for invalid enum value
            filtered_tasks = [t for t in filtered_tasks if t.status == status_filter]
        if assigned_filter:
            filtered_tasks = [t for t in filtered_tasks if t.assigned_to == assigned_filter]
        if orchestration_id_filter: # Apply the new filter
            filtered_tasks = [t for t in filtered_tasks if t.orchestration_id == orchestration_id_filter]

        return [t.model_copy(deep=True) for t in filtered_tasks] # Return copies

    async def _update_task_impl(self, task_id: str, status: Optional[TaskStatus] = None, details: Optional[str] = None, result: Optional[Any] = None, assigned_to: Optional[str] = None) -> Optional[TaskSchema]:
        """Internal method to update specific fields of a task in cache and save."""
        if not self._initialized: await self.initialize()
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"[TaskManager] Attempted to update non-existent task in cache: {task_id}")
                # Optionally: Could try reloading from file? For now, fail if not in cache.
                return None

            updated = False
            now = datetime.now(timezone.utc)

            # Update status (handle string input)
            if status is not None:
                status_enum = None
                if isinstance(status, str):
                    try:
                        status_enum = TaskStatus(status)
                    except ValueError:
                        logger.error(f"[TaskManager] Invalid status value '{status}' for update on task {task_id}.")
                        return None # Or raise error depending on desired strictness
                elif isinstance(status, TaskStatus):
                    status_enum = status
                else:
                    logger.error(f"[TaskManager] Invalid type for status update on task {task_id}: {type(status)}")
                    return None # Or raise error

                if status_enum is not None and task.status != status_enum:
                    task.status = status_enum
                    logger.info(f"[TaskManager] Updated Task ID: {task_id}, Status -> {status_enum.value}")
                    updated = True

            # Update other fields if provided
            if details is not None and task.details != details:
                task.details = details
                logger.debug(f"[TaskManager] Updated Task ID: {task_id}, Details updated.")
                updated = True
            if result is not None: # Allow updating result even if same (e.g., reprocessing)
                task.result = result
                logger.debug(f"[TaskManager] Updated Task ID: {task_id}, Result updated.")
                updated = True
            if assigned_to is not None and task.assigned_to != assigned_to:
                task.assigned_to = assigned_to
                logger.info(f"[TaskManager] Updated Task ID: {task_id}, AssignedTo -> {assigned_to}")
                updated = True

            # If any field was changed, update timestamp and save
            if updated:
                task.updated_at = now
                await self._save_tasks() # Persist changes to shared file

            return task.model_copy(deep=True) # Return updated copy

    async def _delete_task_impl(self, task_id: str) -> bool:
        """Internal method to delete a task by its ID from cache and save."""
        if not self._initialized: await self.initialize()
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id] # Delete from cache
                logger.info(f"[TaskManager] Deleted Task ID: {task_id}")
                await self._save_tasks() # Persist deletion to shared file
                return True
            else:
                logger.warning(f"[TaskManager] Attempted to delete non-existent task from cache: {task_id}")
                return False


    # --- Method to Expose Tools for Agents ---
    def get_tools(self) -> List[ToolSchema]:
        """Returns a list of tools (based on CRUD operations) that agents can use."""
        return [
            ToolSchema(
                name="create_task",
                description="Creates a new task with a unique ID and PENDING status. Can optionally assign it to an agent and link it to an orchestration run.", # Updated description
                parameters=[
                    ToolParameter(name="name", description="The name or title of the task.", type="string", required=True),
                    ToolParameter(name="details", description="Optional details or description for the task.", type="string", required=False),
                    ToolParameter(name="assigned_to", description="Optional name of the agent this task is initially assigned to.", type="string", required=False),
                    ToolParameter(name="orchestration_id", description="Optional ID of the orchestration run this task belongs to.", type="string", required=False), # Parameter schema for orchestration ID
                ],
                callable_ref=self._create_task_impl # Reference the internal implementation
            ),
            ToolSchema(
                name="read_task",
                description="Reads and returns the details of a single task by its ID from the local cache.",
                parameters=[
                    ToolParameter(name="task_id", description="The unique ID of the task to read.", type="string", required=True),
                ],
                callable_ref=self._read_task_impl
            ),
            ToolSchema(
                name="read_all_tasks",
                description="Reads and returns a list of tasks from the local cache, optionally filtering by status, assignee, or the orchestration ID they belong to.", # Updated description
                parameters=[
                    ToolParameter(name="status_filter", description=f"Optional status to filter by (e.g., {', '.join(s.value for s in TaskStatus)}).", type="string", required=False),
                    ToolParameter(name="assigned_filter", description="Optional agent name to filter tasks assigned to.", type="string", required=False),
                    ToolParameter(name="orchestration_id_filter", description="Optional orchestration ID to filter tasks by.", type="string", required=False), # Parameter schema for filtering by orchestration ID
                ],
                callable_ref=self._read_all_tasks_impl
            ),
            ToolSchema(
                name="update_task",
                description="Updates one or more fields (status, details, result, assigned_to) of an existing task by its ID in the cache and saves to the shared log.",
                parameters=[
                    ToolParameter(name="task_id", description="The unique ID of the task to update.", type="string", required=True),
                    ToolParameter(name="status", description=f"Optional new status for the task (e.g., {', '.join(s.value for s in TaskStatus)}).", type="string", required=False), # Note: Type is string here, validation happens internally
                    ToolParameter(name="details", description="Optional new details for the task.", type="string", required=False),
                    ToolParameter(name="result", description="Optional result data to store for the task (can be any JSON-serializable type).", type="any", required=False),
                    ToolParameter(name="assigned_to", description="Optional new assignee for the task.", type="string", required=False),
                ],
                callable_ref=self._update_task_impl
            ),
            ToolSchema(
                name="delete_task",
                description="Deletes a task permanently by its ID from the cache and saves the state to the shared log.",
                parameters=[
                    ToolParameter(name="task_id", description="The unique ID of the task to delete.", type="string", required=True),
                ],
                callable_ref=self._delete_task_impl
            ),
        ]


# --- Helper Functions (Cache, File Analysis) ---
async def is_cache_valid() -> bool:
    """Checks if the Markdown cache file exists and is within the expiration time."""
    if not CACHE_FILE.exists():
        logger.debug(f"[Cache] Cache file not found: {CACHE_FILE}")
        return False
    try:
        cache_age = time.time() - CACHE_FILE.stat().st_mtime
        is_valid = cache_age < CACHE_EXPIRATION
        logger.debug(f"[Cache] Cache file {CACHE_FILE}: Age={cache_age:.0f}s, Valid={is_valid} (Expiration={CACHE_EXPIRATION}s)")
        return is_valid
    except OSError as e:
        logger.error(f"[Cache] Error checking cache validity for {CACHE_FILE}: {e}")
        return False

async def save_to_cache(content: str):
    """Saves Markdown content to the cache file."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        async with aiofiles.open(CACHE_FILE, "w", encoding="utf-8") as f:
            await f.write(content)
        logger.info(f"[Cache] Content saved to cache file: {CACHE_FILE}")
    except IOError as e:
        logger.error(f"[Cache] Error writing cache file {CACHE_FILE}: {e}")

async def load_from_cache() -> Optional[str]:
    """Loads Markdown content from the cache file."""
    if not CACHE_FILE.exists():
         logger.debug(f"[Cache] Cache file not found: {CACHE_FILE}")
         return None
    try:
        async with aiofiles.open(CACHE_FILE, "r", encoding="utf-8") as f:
            content = await f.read()
            logger.info(f"[Cache] Loaded content from cache file: {CACHE_FILE}")
            return content
    except IOError as e:
        logger.error(f"[Cache] Error reading cache file {CACHE_FILE}: {e}")
        return None
    except Exception as e:
        logger.error(f"[Cache] Unexpected error loading cache from {CACHE_FILE}: {e}", exc_info=True)
        return None


async def analyze_files(folder: str, log_step_func: Optional[Callable[[str, str, str, Optional[str]], None]] = None) -> List[Dict[str, Any]]:
    """
    Analyzes Python files within a specified folder, extracting basic information
    like module structure, classes, functions, and docstrings.
    Handles import errors gracefully and attempts to exclude common directories.
    Uses the provided log_step_func for detailed logging.
    """
    current_file_path = None
    try:
        current_file_path = Path(__file__).resolve()
        logger.debug(f"[analyze_files] Current script path: {current_file_path}")
    except NameError:
         logger.warning("[analyze_files] Could not determine current file path (__file__ not defined). No files will be excluded based on self.")

    task_name = "System - Analyze Files"
    exclude_msg = f"Excluding self: {current_file_path.name}" if current_file_path else "No self-exclusion rule applied."
    if log_step_func: log_step_func(task_name, StepStatus.START.value, f"Analyzing folder: {folder}, {exclude_msg}", agent_name="System") # Log start

    loop = asyncio.get_event_loop()
    results = []
    file_count = 0
    error_count = 0
    processed_modules = set()
    excluded_count = 0
    project_root = Path(folder).resolve()
    import sys

    # Temporarily add the project root to sys.path to allow relative imports during analysis
    original_sys_path = sys.path[:]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.debug(f"[analyze_files] Temporarily added {project_root} to sys.path for analysis.")

    try:
        # Common directories and file prefixes/suffixes to exclude from analysis
        exclude_dirs = {".", "__", "venv", ".venv", "env", ".env", "node_modules", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist", "docs", "site-packages", "migrations", "tests", "test", ".vscode", ".idea", ".cache"} # Added .cache
        exclude_prefixes = ("test_", "_test")
        exclude_suffixes = ("_test.py", "_spec.py")

        for root, dirs, files in os.walk(project_root, topdown=True):
            root_path = Path(root)
            # Filter directories based on exclude_dirs
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(('.', '_'))]


            for file in files:
                if file.endswith(".py") and not file.startswith(exclude_prefixes) and not file.endswith(exclude_suffixes) and file != "__init__.py": # Exclude init files for now
                    path = root_path / file
                    abs_path = path.resolve()

                    # Exclude the script running the analysis itself
                    if current_file_path and abs_path == current_file_path:
                        excluded_count += 1
                        continue

                    # Exclude files within excluded directories (double check)
                    in_excluded_dir = False
                    for excluded in exclude_dirs:
                        # Check if the file's parent path contains the excluded dir name segment
                        if excluded in path.parent.parts:
                             in_excluded_dir = True
                             break
                    if in_excluded_dir:
                        excluded_count += 1
                        continue


                    file_count += 1
                    info: Dict[str, Any] = {"filename": "unknown"} # Initialize info dict
                    module_name = "unknown"
                    rel_path_str = ""

                    try:
                        # Generate a Python module-style name from the file path relative to the project root
                        rel_path = path.relative_to(project_root)
                        rel_path_str = str(rel_path).replace(os.sep, "/") # Use forward slashes for consistency
                        info["filename"] = rel_path_str

                        if log_step_func: log_step_func(task_name, StepStatus.IN_PROGRESS.value, f"Processing file: {rel_path_str}", agent_name="System")


                        module_parts = list(rel_path.parts)
                        module_parts[-1] = module_parts[-1].replace('.py', '') # Remove .py extension

                        if not module_parts: continue # Should not happen if filename was valid

                        module_name = '.'.join(module_parts)

                        # Avoid reprocessing the same module if encountered multiple times (e.g., symlinks)
                        if module_name in processed_modules:
                            if log_step_func: log_step_func(task_name, StepStatus.SKIP.value, f"Skipping already processed module: {module_name}", agent_name="System")
                            continue
                        processed_modules.add(module_name) # Mark as processed *before* import attempt

                        info["properties"] = [] # List to store info about classes/functions
                        info["error"] = None # Field to store any analysis errors for this file

                        try:
                            # Run the import in a separate thread using the executor to avoid blocking the event loop
                            # and to isolate potential import side effects (though isolation is limited).
                            mod = await loop.run_in_executor(EXECUTOR, importlib.import_module, module_name)

                            # Inspect the imported module for members (classes, functions, etc.)
                            for name, obj in inspect.getmembers(mod):
                                # Basic filtering (e.g., ignore private/dunder members) - can be customized
                                if not name.startswith("_"):
                                    docstring = ""
                                    obj_type = "unknown"
                                    try:
                                        # Attempt to get docstring and type safely
                                        docstring = inspect.getdoc(obj) or ""
                                        obj_type = str(type(obj).__name__)
                                    except Exception as inspect_err:
                                        # Log specific inspection error but don't stop analysis of other members/files
                                        logger.warning(f"[analyze_files] Error inspecting member '{name}' in module {module_name}: {inspect_err}")
                                        docstring = f"[Error inspecting member: {inspect_err}]"
                                        obj_type = "[Error]"

                                    info["properties"].append({"name": name, "type": obj_type, "doc": docstring.strip()})

                        except ModuleNotFoundError as mnfe:
                            error_msg = f"Module not found during import: {module_name} ({mnfe})"
                            info["error"] = error_msg
                            logger.warning(f"[analyze_files] ModuleNotFoundError for {rel_path_str} -> {module_name}: {mnfe}")
                            if log_step_func: log_step_func(task_name, StepStatus.WARN.value, f"File {rel_path_str}: {error_msg}", agent_name="System")
                            error_count += 1
                        except ImportError as imp_err:
                            # More specific logging for general import errors (e.g., missing dependencies)
                            error_msg = f"Import Error ({type(imp_err).__name__}): {imp_err}"
                            info["error"] = error_msg
                            logger.error(f"[analyze_files] ImportError analyzing {rel_path_str} (Module: {module_name}): {imp_err}", exc_info=False) # Keep logs concise
                            if log_step_func: log_step_func(task_name, StepStatus.ERROR.value, f"File {rel_path_str}: {error_msg}", agent_name="System")
                            error_count += 1
                        except Exception as import_err:
                            # Catch-all for other unexpected errors during import or inspection
                            error_msg = f"Import/Inspect Error ({type(import_err).__name__}): {import_err}"
                            info["error"] = error_msg
                            logger.error(f"[analyze_files] Error analyzing {rel_path_str} (Module: {module_name}): {import_err}", exc_info=False)
                            if log_step_func: log_step_func(task_name, StepStatus.ERROR.value, f"File {rel_path_str}: {error_msg}", agent_name="System")
                            error_count += 1

                    except Exception as outer_err:
                        # Handle errors during path processing or module name generation itself
                        if info["filename"] == "unknown": info["filename"] = str(path) # Try to provide path info if available
                        error_msg = f"File Processing Error ({type(outer_err).__name__}): {outer_err}"
                        info["error"] = error_msg
                        logger.error(f"[analyze_files] Error processing file path {path}: {outer_err}", exc_info=False)
                        if log_step_func: log_step_func(task_name, StepStatus.ERROR.value, f"File {path}: {error_msg}", agent_name="System")
                        error_count += 1

                    results.append(info) # Add the analysis result (or error) for this file

    finally:
        # Restore the original sys.path to avoid side effects
        sys.path = original_sys_path
        logger.debug("[analyze_files] Restored original sys.path.")

    # Log summary of the analysis process
    summary_msg = (f"Analysis complete. Files considered: {file_count}. "
                   f"Unique modules processed: {len(processed_modules)}. Errors during processing: {error_count}. "
                   f"Files/Dirs excluded (self, tests, common dirs): {excluded_count}.")
    if log_step_func: log_step_func(task_name, StepStatus.END.value, summary_msg, agent_name="System") # Log end
    else: logger.info(f"[analyze_files] {summary_msg}")

    return results


# --- Dynamic Refinement (using LLM for iterative improvement) ---
async def dynamic_refine(
    client: GeminiModelClient, # Expects an instance of the custom GeminiClient
    initial_content: str,
    revision_instruction: str, # The goal for refinement
    task_name: str, # Specific sub-task name for context (e.g., "AnalysisRefinement")
    log_step_func: Optional[Callable[[str, str, str, Optional[str]], None]] = None, # Callback for logging steps
    agent_name: str = "DynamicRefine", # Identifier for logging
    # Pass individual generation parameters instead of GenerationConfig/SafetySettings
    # These will be passed to the custom client's generate_content method
    system_instruction: str = "", # System prompt for the LLM
    temperature: float = 0.7,
    top_p: float = 0.95,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> str:
    """
    Performs iterative refinement of provided content using LLM-based revision
    and self-evaluation steps. Uses the custom GeminiClient via a temporary agent.

    Requires BaseAgent to be defined before this function is called.
    """
    # Ensure BaseAgent is available (should be due to script order)
    if 'BaseAgent' not in globals():
         logger.error("[dynamic_refine] CRITICAL: BaseAgent class not defined before dynamic_refine function. Cannot proceed.")
         raise NameError("BaseAgent class is required for dynamic_refine.")

    content = initial_content # Start with the initial content
    action_prefix = f"{agent_name} - Refine-{task_name}" # Base name for logging actions
    logger.info(f"[{action_prefix}] Starting dynamic refinement loop for '{task_name}'. Max iterations: {MAX_ITERATIONS}")
    start_time_refine = time.monotonic()

    # --- Client Validation ---
    # Ensure a valid custom GeminiClient instance was provided
    if not client or not isinstance(client, GeminiClient):
         error_msg = f"[{action_prefix}] Invalid or missing custom GeminiClient passed to dynamic_refine. Cannot proceed."
         logger.error(error_msg)
         # Log this step if callback is available
         if log_step_func: log_step_func(action_prefix, StepStatus.ERROR.value, "Invalid LLM client provided.", agent_name=agent_name)
         raise ValueError(error_msg) # Raise error as refinement is impossible

    # --- Temporary Agent for LLM Calls ---
    # Create a temporary BaseAgent instance solely for using its _execute_llm_call method within this function.
    # This avoids duplicating LLM call logic here. Pass the provided client instance.
    temp_agent_for_llm = BaseAgent(client, tools=None, name=f"{agent_name}-TempRefiner", role="Refiner")
    # Set the default generation parameters on the temporary agent to match the function arguments
    # This ensures _execute_llm_call uses the correct settings unless overridden per call.
    temp_agent_for_llm.default_system_instruction = system_instruction
    temp_agent_for_llm.default_temperature = temperature
    temp_agent_for_llm.default_top_p = top_p
    temp_agent_for_llm.default_frequency_penalty = frequency_penalty
    temp_agent_for_llm.default_presence_penalty = presence_penalty
    logger.debug(f"[{action_prefix}] Temporary refinement agent created with provided generation parameters.")

    # --- Refinement Loop ---
    for i in range(MAX_ITERATIONS):
        iteration = i + 1
        iteration_action_prefix = f"{action_prefix}-Iter{iteration}" # Log prefix for this specific iteration
        logger.debug(f"[{iteration_action_prefix}] Starting iteration {iteration}/{MAX_ITERATIONS}")
        start_time_iter = time.monotonic()

        # --- Revision Step ---
        revise_action = f"{iteration_action_prefix}-Revise"
        # Construct the prompt for the LLM to revise the content
        prompt = (
            f"{revision_instruction}\n\n"
            f"Previous result (Content to improve - Iteration {i}):\n"
            f"```markdown\n{content}\n```\n\n"
            f"Based *only* on the instruction and the previous result provided above, generate the improved content. "
            f"Respond with *only* the full, revised markdown content. Do not include conversational text or explanations."
        )
        revised = "" # Initialize revised content for this iteration
        try:
            step_msg = f"Requesting LLM revision (Iter {iteration})"
            if log_step_func: log_step_func(revise_action, StepStatus.START.value, step_msg, agent_name=agent_name)

            # Use the temporary agent's helper method to call the custom LLM client
            # Pass the specific parameters for this revision step
            revised = await temp_agent_for_llm._execute_llm_call(
                prompt=prompt,
                system_instruction=temp_agent_for_llm.default_system_instruction, # Use agent's (i.e., function's) system instruction
                task_description=f"Refinement Iteration {iteration} - Revision",
                log_step_func=log_step_func, # Pass log func down
                action=revise_action, # Pass action name down
                # Pass the generation parameters specified for the refinement task
                temperature=temp_agent_for_llm.default_temperature,
                top_p=temp_agent_for_llm.default_top_p,
                frequency_penalty=temp_agent_for_llm.default_frequency_penalty,
                presence_penalty=temp_agent_for_llm.default_presence_penalty
            )

            # Handle empty response from LLM
            if not revised or revised.strip() == "":
                 logger.warning(f"[{revise_action}] LLM returned empty content during revision (Iter {iteration}). Using previous content for evaluation.")
                 if log_step_func: log_step_func(revise_action, StepStatus.WARN.value, "LLM returned empty revision.", agent_name=agent_name)
                 # Keep the previous content; let evaluation decide if it's acceptable or needs another try
                 revised = content

            if log_step_func: log_step_func(revise_action, StepStatus.END.value, f"Revision received (Iter {iteration}), Length: {len(revised)}", agent_name=agent_name)

        except asyncio.CancelledError:
            if log_step_func: log_step_func(revise_action, StepStatus.CANCELLED.value, f"Revision cancelled (Iter {iteration})", agent_name=agent_name)
            raise # Propagate cancellation upwards
        except Exception as e:
             logger.error(f"[{revise_action}] Error during LLM revision call (Iter {iteration}): {e}", exc_info=True)
             if log_step_func: log_step_func(revise_action, StepStatus.ERROR.value, f"Revision error (Iter {iteration}): {e}", agent_name=agent_name)
             # Policy: On revision error, return the last successfully generated 'content' to avoid losing progress.
             final_msg = f"Error during revision ({e}). Returning last valid content before this iteration."
             if log_step_func: log_step_func(action_prefix, StepStatus.ERROR.value, final_msg, agent_name=agent_name)
             logger.warning(f"[{action_prefix}] {final_msg}")
             return content # Return the content from the *previous* successful iteration (or initial)

        # --- Evaluation Step ---
        eval_action = f"{iteration_action_prefix}-Evaluate"
        # Construct the prompt for the LLM to evaluate its own revision
        eval_prompt = (
            f"You are an evaluator. Assess if the following generated content requires further revision based on a specific goal.\n"
            f"Goal: '{revision_instruction}'\n"
            f"Task Name: '{task_name}'\n\n"
            f"Current Content:\n"
            f"```markdown\n{revised}\n```\n\n"
            f"Does this content adequately meet the stated goal and require **no more significant revisions**? "
            f"Consider clarity, completeness relative to the goal, and adherence to instructions. "
            f"Answer **strictly** with only 'yes' or 'no'." # Emphasize strict 'yes'/'no' output
        )
        try:
            step_msg = f"Requesting LLM evaluation (Iter {iteration})"
            if log_step_func: log_step_func(eval_action, StepStatus.START.value, step_msg, agent_name=agent_name)

            # Use a simpler LLM configuration for evaluation (low temperature for deterministic yes/no)
            response = await temp_agent_for_llm._execute_llm_call(
                prompt=eval_prompt,
                system_instruction="", # No specific system instruction needed for simple evaluation
                task_description=f"Refinement Iteration {iteration} - Evaluation",
                log_step_func=log_step_func, # Pass log func
                action=eval_action, # Pass action name
                temperature=0.1, # Very low temperature for stricter yes/no output
                top_p=0.9,       # Keep top_p reasonable to avoid overly limiting
                frequency_penalty=0.0, # No penalty needed
                presence_penalty=0.0
            )
            # Clean the response for comparison
            response_cleaned = response.strip().lower() if response else ""

            # Handle empty evaluation response
            if not response_cleaned:
                logger.warning(f"[{eval_action}] Received empty evaluation response (Iter {iteration}). Assuming revision *not* needed (conservative fallback).")
                if log_step_func: log_step_func(eval_action, StepStatus.WARN.value, "Empty evaluation response - assuming 'yes' (passed)", agent_name=agent_name)
                # Treat empty eval as passing to prevent potential infinite loops if evaluation consistently fails. Return the revised content.
                duration_iter_ms = int((time.monotonic() - start_time_iter) * 1000)
                if log_step_func: log_step_func(iteration_action_prefix, StepStatus.END.value, f"Iteration {iteration} complete (Passed on empty eval).", agent_name=agent_name, duration_ms=duration_iter_ms)
                return revised

            # Check evaluation result
            elif response_cleaned.startswith("yes"):
                # Evaluation passed, refinement is complete for this step
                logger.info(f"[{eval_action}] Evaluation passed ('yes') - no further revision needed (Iter {iteration}).")
                if log_step_func: log_step_func(eval_action, StepStatus.END.value, "Evaluation passed ('yes')", agent_name=agent_name)
                duration_iter_ms = int((time.monotonic() - start_time_iter) * 1000)
                if log_step_func: log_step_func(iteration_action_prefix, StepStatus.END.value, f"Iteration {iteration} complete (Passed).", agent_name=agent_name, duration_ms=duration_iter_ms)
                return revised # Return the successfully revised and evaluated content

            elif response_cleaned.startswith("no"):
                 # Evaluation indicates more revision needed
                 content = revised # Update 'content' with the latest revision for the next iteration
                 logger.info(f"[{eval_action}] Evaluation requires further revision ('no') (Iter {iteration}). Continuing loop.")
                 if log_step_func: log_step_func(eval_action, StepStatus.END.value, f"Evaluation requires revision ('no')", agent_name=agent_name)
                 # Continue to the next iteration

            else:
                # Handle ambiguous or unexpected evaluation response
                logger.warning(f"[{eval_action}] Received ambiguous evaluation response: '{response_cleaned}' (Iter {iteration}). Assuming revision needed (conservative).")
                if log_step_func: log_step_func(eval_action, StepStatus.WARN.value, f"Ambiguous evaluation ('{response_cleaned}') - assuming 'no'", agent_name=agent_name)
                content = revised # Update content and continue loop, treating ambiguity as 'no'
                # Continue to the next iteration

            # Log iteration end only if continuing
            if not response_cleaned.startswith("yes"):
                duration_iter_ms = int((time.monotonic() - start_time_iter) * 1000)
                if log_step_func: log_step_func(iteration_action_prefix, StepStatus.END.value, f"Iteration {iteration} complete (Needs more revision).", agent_name=agent_name, duration_ms=duration_iter_ms)


        except asyncio.CancelledError:
            if log_step_func: log_step_func(eval_action, StepStatus.CANCELLED.value, f"Evaluation cancelled (Iter {iteration})", agent_name=agent_name)
            raise # Propagate cancellation
        except Exception as e:
             logger.error(f"[{eval_action}] Error during LLM evaluation call (Iter {iteration}): {e}", exc_info=True)
             if log_step_func: log_step_func(eval_action, StepStatus.ERROR.value, f"Evaluation error (Iter {iteration}): {e}", agent_name=agent_name)
             # Policy: If evaluation fails, it's safer to return the content from the last *successful revision* step.
             final_msg = f"Error during evaluation ({e}). Returning last revised content from this iteration."
             if log_step_func: log_step_func(action_prefix, StepStatus.ERROR.value, final_msg, agent_name=agent_name)
             logger.warning(f"[{action_prefix}] {final_msg}")
             return revised # Return the 'revised' content from the current iteration's revision step

    # --- Loop End ---
    # If the loop finishes without the evaluation passing (reached MAX_ITERATIONS)
    duration_refine_ms = int((time.monotonic() - start_time_refine) * 1000)
    final_msg = f"Reached max iterations ({MAX_ITERATIONS}) without passing evaluation for task '{task_name}'. Returning the last revised content."
    logger.warning(f"[{action_prefix}] {final_msg}")
    if log_step_func: log_step_func(action_prefix, StepStatus.WARN.value, final_msg, agent_name=agent_name, duration_ms=duration_refine_ms)
    return content # Return the last version of content produced


# --- Agent Classes (Base, Planner, Crew) ---
class BaseAgent:
    """
    Base class for all agents, providing core functionalities:
    - LLM interaction (via custom client)
    - Tool usage mechanism
    - Memory management (history, scratchpad) - IN RAM ONLY
    - Storage of defining characteristics (role, goal, backstory)
    - Default generation parameter storage
    """
    def __init__(
        self,
        client: Optional[GeminiModelClient],
        tools: Optional[List[ToolSchema]] = None,
        name: str = "Generic Agent",
        role: str = "Base",
        goal: Optional[str] = None, # Added goal
        backstory: Optional[str] = None # Added backstory
    ):
        # Allow client to be None, but perform checks before using it
        if client is not None and not isinstance(client, GeminiClient):
             logger.error(f"[{name}] CRITICAL: Received incorrect client type: {type(client)}. Expected an instance of the imported GeminiClient or None.")
             raise TypeError(f"Agent '{name}' requires a valid instance of the custom GeminiClient (from app.llm.gemini) or None, but received {type(client)}.")
        self.client = client # Store the client instance (can be None)
        # Create a dictionary for quick tool lookup by name
        self.tools: Dict[str, ToolSchema] = {tool.name: tool for tool in tools} if tools else {}
        self.name = name
        self.role = role
        self.goal = goal # Store goal
        self.backstory = backstory # Store backstory
        # Initialize agent's memory using the Pydantic schema - THIS IS IN RAM
        self.memory = MemorySchema(agent_name=name, agent_role=role)

        # --- Default Generation Parameters ---
        self.default_system_instruction: str = f"You are {self.name}, an AI agent with the role of {self.role}. Your goal is: {self.goal or 'Not specified'}." # Include goal in default system prompt if available
        self.default_temperature: float = 0.7
        self.default_top_p: float = 0.95
        self.default_frequency_penalty: float = 0.0
        self.default_presence_penalty: float = 0.0

        # Log agent initialization details
        tools_list_str = list(self.tools.keys()) if self.tools else 'None'
        client_status = "with custom GeminiClient" if self.client else "WITHOUT LLM client"
        logger.info(f"[{self.name}] Initialized. Role='{self.role}', Goal='{self.goal or 'N/A'}', Backstory='{(self.backstory or 'N/A')[:50]}...', Status='{client_status}', Tools={len(self.tools)}: {tools_list_str}")
        if not self.client:
            logger.warning(f"[{self.name}] Agent initialized without an LLM client. LLM-dependent operations will fail.")


    async def use_tool(self, tool_name: str, log_step_func: Optional[Callable[[str, str, str, Optional[str]], None]] = None, **kwargs) -> Any:
        """
        Executes a specified tool by name with provided keyword arguments.
        Performs validation against the tool's defined parameters.
        Handles both synchronous and asynchronous tool functions.
        Logs step start/end/error using the provided callback.
        """
        action_name = f"{self.name} - UseTool-{tool_name}" # Action name for logging
        tool = self.tools.get(tool_name)
        start_time_tool = time.monotonic()

        # --- Tool Validation ---
        if not tool:
            error_msg = f"Tool '{tool_name}' not found for agent '{self.name}'. Available tools: {list(self.tools.keys())}"
            logger.error(f"[{action_name}] {error_msg}")
            self.memory.add_history("tool_error", {"tool_name": tool_name, "error": "Tool not found", "available": list(self.tools.keys()), "args": kwargs})
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, error_msg, agent_name=self.name)
            raise ValueError(error_msg) # Raise error as tool cannot be used

        if log_step_func: log_step_func(action_name, StepStatus.START.value, f"Attempting to use tool '{tool_name}' with args: {kwargs}", agent_name=self.name)


        # --- Argument Validation ---
        provided_args = set(kwargs.keys())
        required_params = {p.name for p in tool.parameters if p.required}
        allowed_params = {p.name for p in tool.parameters}

        # Check for missing required arguments
        missing_required = required_params - provided_args
        if missing_required:
             error_msg = f"Missing required arguments for tool '{tool_name}': {missing_required}. Provided: {list(provided_args)}"
             logger.error(f"[{action_name}] {error_msg}")
             self.memory.add_history("tool_error", {"tool_name": tool_name, "error": "Missing required arguments", "missing": list(missing_required), "provided": list(provided_args)})
             if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, error_msg, agent_name=self.name)
             raise ValueError(error_msg)

        # Check for extra/unknown arguments (optional: warn and filter)
        invalid_args = provided_args - allowed_params
        if invalid_args:
             warning_msg = f"Ignoring unexpected arguments provided for tool '{tool_name}': {invalid_args}. Allowed parameters: {list(allowed_params)}"
             logger.warning(f"[{action_name}] {warning_msg}")
             if log_step_func: log_step_func(action_name, StepStatus.WARN.value, warning_msg, agent_name=self.name)
             kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}

        # --- Type Handling (Example: TaskStatus Enum) ---
        status_arg_name = None
        if tool_name == "update_task" and "status" in kwargs: status_arg_name = "status"
        elif tool_name == "read_all_tasks" and "status_filter" in kwargs: status_arg_name = "status_filter"

        if status_arg_name and isinstance(kwargs[status_arg_name], str):
            status_str = kwargs[status_arg_name]
            try:
                kwargs[status_arg_name] = TaskStatus(status_str)
                logger.debug(f"[{action_name}] Converted string '{status_str}' to TaskStatus enum for argument '{status_arg_name}'.")
            except ValueError:
                 valid_statuses = ', '.join(s.value for s in TaskStatus)
                 error_msg = f"Invalid status value '{status_str}' provided for argument '{status_arg_name}' in tool '{tool_name}'. Valid values are: {valid_statuses}"
                 logger.error(f"[{action_name}] {error_msg}")
                 self.memory.add_history("tool_error", {"tool_name": tool_name, "error": "Invalid enum value", "argument": status_arg_name, "value": status_str, "valid": valid_statuses})
                 if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, error_msg, agent_name=self.name)
                 raise ValueError(error_msg) # Raise error for invalid input

        # --- Tool Execution ---
        log_kwargs = {k: (v.value if isinstance(v, Enum) else v) for k, v in kwargs.items()}
        logger.info(f"[{action_name}] Executing Tool: '{tool_name}' with validated args: {log_kwargs}")
        self.memory.add_history("tool_use_start", {"tool_name": tool_name, "args": log_kwargs})

        try:
            callable_func = tool.callable_ref
            if asyncio.iscoroutinefunction(callable_func):
                result = await callable_func(**kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(EXECUTOR, lambda: callable_func(**kwargs))

            # --- Result Handling & Logging ---
            duration_tool_ms = int((time.monotonic() - start_time_tool) * 1000)
            result_summary = f"Type: {type(result).__name__}"
            try: # Safely try to add more detail to the summary
                 if isinstance(result, list): result_summary += f", Count: {len(result)}"
                 elif isinstance(result, (str, bytes)): result_summary += f", Length: {len(result)}"
                 elif isinstance(result, BaseModel):
                     summary_json = result.model_dump_json(indent=None, exclude_none=True)[:150] # Limit length
                     result_summary = f"Pydantic Model: {summary_json}" + ("..." if len(result.model_dump_json()) > 150 else "")
                 elif result is None: result_summary = "None"
            except Exception as summary_err:
                 result_summary += f" (Error summarizing result: {summary_err})"

            logger.info(f"[{action_name}] Tool '{tool_name}' executed successfully. Result summary: {result_summary}")
            self.memory.add_history("tool_use_end", {"tool_name": tool_name, "success": True, "result_summary": result_summary})
            if log_step_func: log_step_func(action_name, StepStatus.END.value, f"Tool '{tool_name}' executed successfully. Result: {result_summary}", agent_name=self.name, duration_ms=duration_tool_ms)
            self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
            return result # Return the actual result from the tool

        except Exception as e:
            duration_tool_ms = int((time.monotonic() - start_time_tool) * 1000)
            error_msg = f"Error executing tool '{tool_name}': {type(e).__name__}: {e}"
            logger.error(f"[{action_name}] {error_msg}", exc_info=True) # Log full traceback for tool execution errors
            self.memory.add_history("tool_use_end", {"tool_name": tool_name, "success": False, "error": error_msg})
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, error_msg, agent_name=self.name, duration_ms=duration_tool_ms)
            self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
            raise # Re-raise the exception


    async def _execute_llm_call(
        self,
        prompt: str,
        task_description: str, # Description of the task for logging
        log_step_func: Optional[Callable[[str, str, str, Optional[str], Optional[int]], None]] = None, # Updated signature
        action: str = "GenerateContent", # Specific action name (e.g., "GenerateAnalysis-Initial")
        # --- Generation Parameters ---
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None
    ) -> str:
        """
        Internal helper method to execute a call to the custom GeminiClient's
        `generate_content` method. Handles client checking, parameter resolution,
        execution, logging (including step logging), timing, and error handling.
        """
        action_name = f"{self.name} - {action}" # Construct full action name for logging
        logger.debug(f"[{action_name}] Preparing LLM call for task: '{task_description}'")
        self.memory.add_history("llm_call_start", {"action": action, "description": task_description})
        if log_step_func: log_step_func(action_name, StepStatus.START.value, task_description, agent_name=self.name)

        # --- Client Check ---
        if not self.client or not isinstance(self.client, GeminiClient):
             error_msg = f"LLM client (custom GeminiClient) is not available or not configured for agent '{self.name}'. Cannot execute LLM call for '{task_description}'."
             logger.error(f"[{action_name}] {error_msg}")
             self.memory.add_history("llm_call_end", {"action": action, "success": False, "status": "Error", "error": "LLM client unavailable", "duration": 0})
             if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, "LLM client unavailable.", agent_name=self.name)
             self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
             raise RuntimeError(error_msg)

        # --- Parameter Resolution ---
        # Resolve system instruction: Use provided, then agent default, then generic default
        current_system_instruction = system_instruction if system_instruction is not None else self.default_system_instruction
        if not current_system_instruction: # Fallback if even default is empty
             current_system_instruction = f"You are {self.name}, an AI agent with the role of {self.role}."

        current_temperature = temperature if temperature is not None else self.default_temperature
        current_top_p = top_p if top_p is not None else self.default_top_p
        current_frequency_penalty = frequency_penalty if frequency_penalty is not None else self.default_frequency_penalty
        current_presence_penalty = presence_penalty if presence_penalty is not None else self.default_presence_penalty
        logger.debug(f"[{action_name}] Using parameters: Temp={current_temperature}, TopP={current_top_p}, FreqPen={current_frequency_penalty}, PresPen={current_presence_penalty}")
        logger.debug(f"[{action_name}] Using System Instruction: '{current_system_instruction[:100]}...'")

        # --- LLM Call Execution ---
        start_time = time.monotonic()
        response_text = "" # Initialize response variable

        try:
            loop = asyncio.get_event_loop()
            if not hasattr(self.client, 'generate_content') or not callable(self.client.generate_content):
                 raise AttributeError(f"The configured GeminiClient instance for agent '{self.name}' does not have a callable 'generate_content' method.")

            response_text = await loop.run_in_executor(
                EXECUTOR, # Use the shared thread pool
                self.client.generate_content, # The synchronous method to call on the client instance
                prompt,
                current_system_instruction,
                current_temperature,
                current_top_p,
                current_frequency_penalty,
                current_presence_penalty
            )
            duration = time.monotonic() - start_time
            duration_ms = int(duration * 1000)

            # --- Process Successful Response ---
            if isinstance(response_text, str):
                 response_length = len(response_text)
                 logger.debug(f"[{action_name}] LLM call successful. Duration: {duration:.2f}s. Response length: {response_length}")
                 self.memory.add_history("llm_call_end", {"action": action, "success": True, "duration": duration, "response_length": response_length})
                 if log_step_func: log_step_func(action_name, StepStatus.END.value, f"Success ({duration:.2f}s), Len: {response_length}", agent_name=self.name, duration_ms=duration_ms)
                 self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
                 return response_text.strip()
            else:
                 error_msg = f"LLM call returned unexpected type: {type(response_text)}. Expected str."
                 logger.error(f"[{action_name}] {error_msg}")
                 self.memory.add_history("llm_call_end", {"action": action, "success": False, "status": "Error", "error": "Unexpected response type", "duration": duration_ms})
                 if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, "Unexpected response type from LLM client.", agent_name=self.name, duration_ms=duration_ms)
                 self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
                 return ""

        except asyncio.CancelledError:
            duration = time.monotonic() - start_time
            duration_ms = int(duration * 1000)
            logger.warning(f"[{action_name}] LLM call cancelled after {duration:.2f}s")
            self.memory.add_history("llm_call_end", {"action": action, "success": False, "status": "Cancelled", "duration": duration_ms})
            if log_step_func: log_step_func(action_name, StepStatus.CANCELLED.value, f"Cancelled after {duration:.2f}s", agent_name=self.name, duration_ms=duration_ms)
            self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
            raise

        except Exception as e:
            duration = time.monotonic() - start_time
            duration_ms = int(duration * 1000)
            error_type = type(e).__name__
            error_msg = f"Error during LLM call for '{task_description}' after {duration:.2f}s: {error_type}: {e}"
            logger.error(f"[{action_name}] {error_msg}", exc_info=True)
            self.memory.add_history("llm_call_end", {"action": action, "success": False, "status": "Error", "error": f"{error_type}: {e}", "duration": duration_ms})
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, f"Error after {duration:.2f}s: {error_type}", agent_name=self.name, duration_ms=duration_ms)
            self.memory.last_updated = datetime.now(timezone.utc) # Update memory timestamp
            raise e


    async def react_loop(
        self,
        initial_content: str, # The starting content to be refined
        revision_instruction: str, # The goal for refinement
        task_id_to_update: str, # The ID of the TaskManager task associated with this refinement
        refine_subtask_name: str, # A specific name for this refinement step (e.g., "AnalysisRefinement")
        log_step_func: Optional[Callable[[str, str, str, Optional[str]], None]] = None, # Step logging callback
        # --- Generation Parameters (Optional Overrides) ---
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> str:
        """
        Orchestrates the dynamic refinement process (`dynamic_refine`) for a specific
        piece of content generated by an agent. Updates tasks and logs steps.
        Updates agent's in-memory `current_task_id`.
        """
        react_action_name = f"{self.name} - ReactLoop-{refine_subtask_name}" # Logging prefix
        logger.info(f"[{react_action_name}] Starting refinement loop for content associated with task {task_id_to_update}.")
        start_time_react = time.monotonic()
        if log_step_func: log_step_func(react_action_name, StepStatus.START.value, f"Starting refinement for task {task_id_to_update}", agent_name=self.name)


        # --- Pre-check: LLM Client Availability ---
        if not self.client:
             error_msg = f"[{react_action_name}] Cannot start refinement: Agent '{self.name}' has no LLM client configured."
             logger.error(error_msg)
             if log_step_func: log_step_func(react_action_name, StepStatus.ERROR.value, "Agent missing LLM client.", agent_name=self.name)
             try:
                 await self.use_tool( "update_task", task_id=task_id_to_update, status=TaskStatus.ERROR, details=f"Refinement failed: Agent '{self.name}' missing required LLM client.", log_step_func=log_step_func )
                 logger.info(f"[{react_action_name}] Marked task {task_id_to_update} as ERROR due to missing LLM client.")
             except Exception as update_err:
                 logger.error(f"[{react_action_name}] CRITICAL: Failed to update task {task_id_to_update} to ERROR status after discovering missing LLM client: {update_err}")
             raise RuntimeError(error_msg)

        # --- Task Status Update: Start Refinement ---
        # Set current task ID in agent's in-memory state
        self.memory.current_task_id = task_id_to_update
        self.memory.last_updated = datetime.now(timezone.utc)
        try:
            await self.use_tool(
                "update_task",
                task_id=task_id_to_update,
                status=TaskStatus.IN_PROGRESS,
                details=f"Starting refinement step: {refine_subtask_name}",
                log_step_func=log_step_func # Pass log func to tool use
            )
            logger.info(f"[{react_action_name}] Marked task {task_id_to_update} as IN_PROGRESS.")
        except Exception as e:
             logger.error(f"[{react_action_name}] CRITICAL: Failed to update task {task_id_to_update} to IN_PROGRESS before starting refinement: {e}", exc_info=True)
             if log_step_func: log_step_func(react_action_name, StepStatus.ERROR.value, f"Failed to set task {task_id_to_update} IN_PROGRESS: {e}", agent_name=self.name)
             # Clear current task ID in agent's memory on failure
             self.memory.current_task_id = None
             self.memory.last_updated = datetime.now(timezone.utc)
             raise

        # --- Execute Dynamic Refinement ---
        try:
            # Resolve generation parameters for dynamic_refine, using passed values or agent defaults
            refine_system_instruction = system_instruction if system_instruction is not None else self.default_system_instruction
            refine_temperature = temperature if temperature is not None else self.default_temperature
            refine_top_p = top_p if top_p is not None else self.default_top_p
            refine_frequency_penalty = frequency_penalty if frequency_penalty is not None else self.default_frequency_penalty
            refine_presence_penalty = presence_penalty if presence_penalty is not None else self.default_presence_penalty

            revised_content = await dynamic_refine(
                client=self.client,
                initial_content=initial_content,
                revision_instruction=revision_instruction,
                task_name=refine_subtask_name,
                log_step_func=log_step_func, # Pass log func down
                agent_name=self.name,
                system_instruction=refine_system_instruction,
                temperature=refine_temperature,
                top_p=refine_top_p,
                frequency_penalty=refine_frequency_penalty,
                presence_penalty=refine_presence_penalty,
            )

            # --- Task Status Update: Successful Completion ---
            duration_react_ms = int((time.monotonic() - start_time_react) * 1000)
            logger.info(f"[{react_action_name}] Dynamic refinement completed successfully for task {task_id_to_update}. Storing result.")
            if log_step_func: log_step_func(react_action_name, StepStatus.END.value, f"Refinement successful for task {task_id_to_update}", agent_name=self.name, duration_ms=duration_react_ms)

            await self.use_tool(
                "update_task",
                task_id=task_id_to_update,
                status=TaskStatus.DONE, # Mark task as done
                details=f"Refinement completed successfully: {refine_subtask_name}",
                result=revised_content, # Store the final content
                log_step_func=log_step_func # Pass log func
            )
            logger.info(f"[{react_action_name}] Marked task {task_id_to_update} as DONE.")
            # Clear current task ID in agent's memory upon completion
            self.memory.current_task_id = None
            self.memory.last_updated = datetime.now(timezone.utc)
            return revised_content

        # --- Error & Cancellation Handling ---
        except asyncio.CancelledError:
            duration_react_ms = int((time.monotonic() - start_time_react) * 1000)
            logger.warning(f"[{react_action_name}] React loop refinement was cancelled for task {task_id_to_update}.")
            if log_step_func: log_step_func(react_action_name, StepStatus.CANCELLED.value, f"Refinement cancelled for task {task_id_to_update}", agent_name=self.name, duration_ms=duration_react_ms)
            try:
                current_task = await self.use_tool("read_task", task_id=task_id_to_update, log_step_func=log_step_func)
                if current_task and current_task.status not in [TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED]:
                     await self.use_tool( "update_task", task_id=task_id_to_update, status=TaskStatus.CANCELLED, details=f"Refinement cancelled during: {refine_subtask_name}", log_step_func=log_step_func )
                     logger.info(f"[{react_action_name}] Marked task {task_id_to_update} as CANCELLED.")
                elif current_task:
                     logger.info(f"[{react_action_name}] Task {task_id_to_update} was already in terminal state ({current_task.status.value}) upon cancellation.")
                else:
                     logger.warning(f"[{react_action_name}] Task {task_id_to_update} not found when trying to mark as cancelled.")
            except Exception as update_err:
                logger.error(f"[{react_action_name}] Failed to update task {task_id_to_update} to CANCELLED status after cancellation event: {update_err}")
            finally:
                 # Clear current task ID in agent's memory after cancellation
                 if self.memory.current_task_id == task_id_to_update:
                    self.memory.current_task_id = None
                    self.memory.last_updated = datetime.now(timezone.utc)
                 raise

        except Exception as e:
             duration_react_ms = int((time.monotonic() - start_time_react) * 1000)
             error_type = type(e).__name__
             logger.error(f"[{react_action_name}] Error during react loop refinement for task {task_id_to_update}: {error_type}: {e}", exc_info=True)
             if log_step_func: log_step_func(react_action_name, StepStatus.ERROR.value, f"Error during refinement: {error_type}: {e}", agent_name=self.name, duration_ms=duration_react_ms)
             try:
                 current_task = await self.use_tool("read_task", task_id=task_id_to_update, log_step_func=log_step_func)
                 if current_task and current_task.status not in [TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED]:
                     await self.use_tool( "update_task", task_id=task_id_to_update, status=TaskStatus.ERROR, details=f"Error during refinement step '{refine_subtask_name}': {error_type}: {str(e)[:200]}", log_step_func=log_step_func )
                     logger.info(f"[{react_action_name}] Marked task {task_id_to_update} as ERROR.")
                 elif current_task:
                    logger.info(f"[{react_action_name}] Task {task_id_to_update} was already in terminal state ({current_task.status.value}) upon error.")
                 else:
                     logger.warning(f"[{react_action_name}] Task {task_id_to_update} not found when trying to mark as error.")
             except Exception as update_err:
                 logger.error(f"[{react_action_name}] CRITICAL: Failed to update task {task_id_to_update} to ERROR status after exception: {update_err}")
             finally:
                  # Clear current task ID in agent's memory after error
                  if self.memory.current_task_id == task_id_to_update:
                     self.memory.current_task_id = None
                     self.memory.last_updated = datetime.now(timezone.utc)
                  raise


class PlannerAgent(BaseAgent):
    """
    Agent responsible for initial planning and analysis tasks.
    Specifically, analyzes the file structure information provided.
    """
    def __init__(self, client: Optional[GeminiModelClient], tools: List[ToolSchema], name="Planner Agent", goal: Optional[str] = None, backstory: Optional[str] = None):
        super().__init__(client, tools, name=name, role="Planner", goal=goal, backstory=backstory) # Pass goal/backstory
        if not self.client:
            logger.warning(f"[{self.name}] Initialized WITHOUT a valid LLM client. Project analysis generation will fail.")


    async def generate_analysis(self, task_id: str, files_info: list, log_step_func=None) -> str:
        """
        Generates a project analysis document based on the output of `analyze_files`.
        Uses the LLM for initial generation and then refines the output using `react_loop`.
        Updates the associated TaskManager task (`task_id`) throughout the process.
        Updates agent's in-memory `current_task_id`.
        """
        action_name = f"{self.name} - GenerateAnalysis"
        task_desc = f"Generate Project Analysis (Task ID: {task_id})"
        logger.info(f"[{action_name}] Starting: {task_desc}")
        if log_step_func: log_step_func(action_name, StepStatus.START.value, task_desc, agent_name=self.name, task_id=task_id)
        start_time_analysis = time.monotonic()


        # --- Pre-check: LLM Client ---
        if not self.client:
            error_msg = f"[{action_name}] Cannot generate analysis for task {task_id}: Agent '{self.name}' is missing the required LLM client."
            logger.error(error_msg)
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, "Agent missing LLM client.", agent_name=self.name, task_id=task_id)
            try:
                await self.use_tool("update_task", task_id=task_id, status=TaskStatus.ERROR, details=error_msg, log_step_func=log_step_func)
                logger.info(f"[{action_name}] Marked task {task_id} as ERROR due to missing client.")
            except Exception as update_err:
                logger.error(f"[{action_name}] Failed to update task {task_id} to ERROR after finding missing client: {update_err}")
            raise RuntimeError(error_msg) # Fail fast

        # Set current task in agent's memory
        self.memory.current_task_id = task_id
        self.memory.last_updated = datetime.now(timezone.utc)

        # --- Update Task Status: Start Analysis ---
        try:
            await self.use_tool("update_task", task_id=task_id, status=TaskStatus.IN_PROGRESS, details="Starting project analysis based on file structure info.", log_step_func=log_step_func)
            logger.info(f"[{action_name}] Marked task {task_id} as IN_PROGRESS.")
        except Exception as e:
             logger.error(f"[{action_name}] CRITICAL: Failed to set task {task_id} to IN_PROGRESS before analysis: {e}", exc_info=True)
             if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, f"Failed to set task IN_PROGRESS: {e}", agent_name=self.name, task_id=task_id)
             if self.memory.current_task_id == task_id:
                 self.memory.current_task_id = None # Clear memory
                 self.memory.last_updated = datetime.now(timezone.utc)
             raise

        analysis_content = "" # Initialize variable to hold the analysis markdown
        try:
            # --- Step 1: Initial Analysis Generation using LLM ---
            files_summary = []
            max_props_per_file = 5 # Limit detail in prompt to avoid excessive length
            max_files_in_prompt = 100 # Limit number of files listed
            files_processed_count = 0
            error_file_count = 0

            for info in files_info:
                if files_processed_count >= max_files_in_prompt:
                    files_summary.append(f"... (truncated after {max_files_in_prompt} files)")
                    break

                props_list = info.get('properties', [])
                props_summary = ", ".join([p.get('name', '?') for p in props_list[:max_props_per_file]])
                if len(props_list) > max_props_per_file: props_summary += ", ..."

                error_info = ""
                if info.get("error"):
                    error_info = f" (Analysis Error: {str(info['error'])[:50]}...)"
                    error_file_count += 1

                files_summary.append(f"- `{info.get('filename', 'N/A')}`:{error_info} Key items: [{props_summary if props_summary else '(No items found/analyzed)'}]")
                files_processed_count += 1

            files_input_str = "\n".join(files_summary) if files_summary else "No Python files suitable for analysis were found or processed."
            if len(files_input_str) > 30000: # Add basic length check for safety
                 logger.warning(f"[{action_name}] Files info summary string is very long ({len(files_input_str)} chars). Truncating for LLM prompt.")
                 files_input_str = files_input_str[:30000] + "\n... (summary truncated)"

            prompt = (
                f"**Role:** You are the {self.name}, specializing in software project analysis.\n"
                f"**Task:** Analyze the following file structure information extracted from a Python project codebase. Your goal is to provide a high-level summary of the project.\n"
                f"**Input Data:** A list of Python files found, key members (like classes/functions) detected within them, and any errors encountered during the automated analysis.\n\n"
                f"**Instructions:**\n"
                f"1.  Review the file information summary below.\n"
                f"2.  Based *only* on this provided information (filenames, member names, errors), infer the project's likely purpose and main components.\n"
                f"3.  Identify key modules or files that seem central to the project's function.\n"
                f"4.  Briefly mention any files where analysis errors occurred, as these might indicate issues or complex code.\n"
                f"5.  Do *not* invent features or details not supported by the input data.\n"
                f"6.  Format your response as a concise Markdown document, starting *only* with a `## Project Analysis` header.\n\n"
                f"**File Information Summary:**\n{files_input_str}\n\n"
                f"**Output:**\n## Project Analysis\n"
            )

            logger.info(f"[{action_name}] Requesting initial analysis generation from LLM.")
            # Use agent's default generation parameters for the initial call
            analysis_content = await self._execute_llm_call(
                prompt=prompt,
                task_description="Generate Initial Project Analysis",
                log_step_func=log_step_func, # Pass log func
                action="GenerateAnalysis-Initial"
                # system_instruction=self.default_system_instruction, # Handled within _execute_llm_call
                # temperature=self.default_temperature,
                # top_p=self.default_top_p,
                # frequency_penalty=self.default_frequency_penalty,
                # presence_penalty=self.default_presence_penalty
            )

            if not analysis_content or analysis_content.strip() == "" or analysis_content.strip().lower() == "## project analysis":
                warning_msg = "LLM returned empty or placeholder initial analysis. Using a basic placeholder and proceeding to refinement."
                logger.warning(f"[{action_name}] {warning_msg}")
                if log_step_func: log_step_func(action_name, StepStatus.WARN.value, warning_msg, agent_name=self.name, task_id=task_id)
                analysis_content = (
                    f"## Project Analysis\n\n"
                    f"*Initial analysis generation based on file info failed or returned empty. Manual review or refinement is required.*\n\n"
                    f"**Summary of Files Analyzed:**\n{files_input_str}\n\n"
                    f"**Analysis Errors Encountered:** {error_file_count} file(s) reported issues during automated analysis."
                 )
                try:
                    await self.use_tool("update_task", task_id=task_id, details="Initial analysis generation failed; attempting refinement with placeholder content.", log_step_func=log_step_func)
                except Exception as update_err:
                     logger.warning(f"[{action_name}] Failed to update task details about initial analysis failure: {update_err}")

            # --- Step 2: Refinement using React Loop ---
            logger.info(f"[{action_name}] Starting refinement of the generated analysis using react_loop.")
            # Pass agent's default generation parameters to react_loop, allowing overrides if needed
            refined_analysis = await self.react_loop(
                initial_content=analysis_content,
                revision_instruction=(
                    "**Refinement Goal:** Review and refine the provided project analysis Markdown.\n"
                    "1.  Ensure the analysis is clear, concise, and accurately reflects the information presented in the initial content (which was based on file structure).\n"
                    "2.  Improve the logical flow and readability.\n"
                    "3.  Verify it starts directly with the `## Project Analysis` header.\n"
                    "4.  Remove any conversational filler text, self-correction statements, or generation artifacts.\n"
                    "5.  Focus *only* on improving the existing content based on these instructions; do not add new analysis points."
                ),
                task_id_to_update=task_id,
                refine_subtask_name="AnalysisRefinement",
                log_step_func=log_step_func, # Pass log func down
                system_instruction=self.default_system_instruction, # Pass defaults
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                frequency_penalty=self.default_frequency_penalty,
                presence_penalty=self.default_presence_penalty
            )

            # If react_loop completed successfully, the task is already marked DONE, and memory.current_task_id cleared.
            duration_analysis_ms = int((time.monotonic() - start_time_analysis) * 1000)
            logger.info(f"[{action_name}] Successfully completed task: {task_desc}")
            if log_step_func: log_step_func(action_name, StepStatus.END.value, f"Successfully completed analysis task {task_id}", agent_name=self.name, task_id=task_id, duration_ms=duration_analysis_ms)
            return refined_analysis

        except asyncio.CancelledError:
             duration_analysis_ms = int((time.monotonic() - start_time_analysis) * 1000)
             logger.warning(f"[{action_name}] Cancelled task: {task_desc}")
             if log_step_func: log_step_func(action_name, StepStatus.CANCELLED.value, f"Analysis task {task_id} cancelled", agent_name=self.name, task_id=task_id, duration_ms=duration_analysis_ms)
             # Ensure task ID is cleared from memory even if update fails
             if self.memory.current_task_id == task_id:
                 self.memory.current_task_id = None
                 self.memory.last_updated = datetime.now(timezone.utc)
                 try:
                     # Check task status before marking cancelled
                     current_task = await self.use_tool("read_task", task_id=task_id, log_step_func=log_step_func)
                     if current_task and current_task.status == TaskStatus.IN_PROGRESS:
                         await self.use_tool("update_task", task_id=task_id, status=TaskStatus.CANCELLED, details="Task cancelled during initial analysis generation.", log_step_func=log_step_func)
                         logger.info(f"[{action_name}] Marked task {task_id} as CANCELLED.")
                 except Exception as update_err:
                      logger.error(f"[{action_name}] Failed to update task {task_id} to CANCELLED after cancellation event: {update_err}")
             raise

        except Exception as e:
            duration_analysis_ms = int((time.monotonic() - start_time_analysis) * 1000)
            error_type = type(e).__name__
            logger.error(f"[{action_name}] Error processing task {task_desc}: {error_type}: {e}", exc_info=True)
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, f"Error in analysis task {task_id}: {error_type}: {e}", agent_name=self.name, task_id=task_id, duration_ms=duration_analysis_ms)
            # Ensure task ID is cleared from memory even if update fails
            if self.memory.current_task_id == task_id:
                 self.memory.current_task_id = None
                 self.memory.last_updated = datetime.now(timezone.utc)
                 try:
                     # Check task status before marking error
                     current_task = await self.use_tool("read_task", task_id=task_id, log_step_func=log_step_func)
                     if current_task and current_task.status == TaskStatus.IN_PROGRESS:
                         await self.use_tool( "update_task", task_id=task_id, status=TaskStatus.ERROR, details=f"Error generating analysis: {error_type}: {str(e)[:200]}", result=analysis_content, log_step_func=log_step_func )
                         logger.info(f"[{action_name}] Marked task {task_id} as ERROR.")
                 except Exception as update_err:
                      logger.error(f"[{action_name}] CRITICAL: Failed to update task {task_id} to ERROR status after exception: {update_err}")
            raise


class CrewAgent(BaseAgent):
    """
    Agent responsible for building upon previous steps in the documentation
    generation process, such as creating structure, enhancing design,
    and adding knowledge base sections.
    """
    def __init__(self, client: Optional[GeminiModelClient], tools: List[ToolSchema], name="Crew Agent", goal: Optional[str] = None, backstory: Optional[str] = None):
        super().__init__(client, tools, name=name, role="Crew", goal=goal, backstory=backstory) # Pass goal/backstory
        if not self.client:
            logger.warning(f"[{self.name}] Initialized WITHOUT a valid LLM client. Document generation steps will fail.")

    async def _check_client_and_set_task_status(self, task_id: str, action_name: str, start_details: str, log_step_func) -> bool:
        """
        Helper method for CrewAgent tasks. Checks for LLM client availability
        and attempts to set the associated task to IN_PROGRESS. Logs steps.
        Updates agent's in-memory `current_task_id`.
        Returns True if successful, False otherwise.
        """
        # Check for LLM Client
        if not self.client:
            error_msg = f"[{action_name}] Cannot start task {task_id}: Agent '{self.name}' is missing the required LLM client."
            logger.error(error_msg)
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, "Agent missing LLM client.", agent_name=self.name, task_id=task_id)
            try:
                await self.use_tool("update_task", task_id=task_id, status=TaskStatus.ERROR, details=error_msg, log_step_func=log_step_func)
                logger.info(f"[{action_name}] Marked task {task_id} as ERROR due to missing client.")
            except Exception as update_err:
                logger.error(f"[{action_name}] Failed to update task {task_id} to ERROR after finding missing client: {update_err}")
            return False # Indicate failure

        # Set current task in agent's memory and update status
        self.memory.current_task_id = task_id
        self.memory.last_updated = datetime.now(timezone.utc)
        logger.info(f"[{action_name}] Starting task (ID: {task_id}) - {start_details}")
        try:
            await self.use_tool("update_task", task_id=task_id, status=TaskStatus.IN_PROGRESS, details=start_details, log_step_func=log_step_func)
            logger.info(f"[{action_name}] Marked task {task_id} as IN_PROGRESS.")
            if log_step_func: log_step_func(action_name, StepStatus.START.value, start_details, agent_name=self.name, task_id=task_id)
            return True # Indicate success
        except Exception as e:
            logger.error(f"[{action_name}] CRITICAL: Failed to set task {task_id} to IN_PROGRESS: {e}", exc_info=True)
            if log_step_func: log_step_func(action_name, StepStatus.ERROR.value, f"Failed to set task IN_PROGRESS: {e}", agent_name=self.name, task_id=task_id)
            # Clear memory if task update fails
            if self.memory.current_task_id == task_id:
                 self.memory.current_task_id = None
                 self.memory.last_updated = datetime.now(timezone.utc)
            return False # Indicate failure

    async def generate_structure(self, task_id: str, analysis: str, log_step_func=None) -> str:
        """
        Generates a basic Markdown document structure (headers) based on the project analysis.
        Uses `react_loop` for refinement and handles task updates. Logs steps.
        Updates agent's in-memory `current_task_id`.
        """
        action_name = f"{self.name} - GenerateStructure"
        start_time_structure = time.monotonic()
        if not await self._check_client_and_set_task_status(task_id, action_name, "Generating document structure from analysis.", log_step_func):
            raise RuntimeError(f"Failed to start structure generation task {task_id} due to missing client or task update error.")

        structure_content = "" # Initialize structure content variable
        try:
            # --- Initial Structure Generation ---
            prompt = (
                f"**Role:** You are the {self.name}, an AI agent skilled in technical documentation.\n"
                f"**Task:** Create a logical Markdown document structure (using level 2 `##` and level 3 `###` headers) based on the provided project analysis. This structure will serve as a template for project documentation (e.g., a README).\n"
                f"**Input:** Project Analysis Markdown.\n\n"
                f"**Instructions:**\n"
                f"1.  Review the analysis below.\n"
                f"2.  Identify the key topics and components mentioned.\n"
                f"3.  Propose a clear sequence of sections (e.g., Overview, Features, Installation, Usage, API/Components, Contributing, License).\n"
                f"4.  Focus on organizing the information logically based *only* on the analysis content.\n"
                f"5.  Start with a suitable top-level header (e.g., `# Project Documentation` or `# [Project Name]`).\n"
                f"6.  Respond *only* with the Markdown structure itself (headers only).\n\n"
                f"**Project Analysis Input:**\n```markdown\n{analysis}\n```\n\n"
                f"**Output:**\n# [Project Name] Documentation\n"
            )
            logger.info(f"[{action_name}] Requesting initial structure generation from LLM.")
            structure_content = await self._execute_llm_call(
                prompt=prompt,
                task_description="Generate Initial Document Structure",
                log_step_func=log_step_func, # Pass log func
                action=f"{action_name}-Initial"
                # Use agent defaults for generation parameters
            )

            if not structure_content or not structure_content.strip() or len(structure_content.strip().split('\n')) < 2:
                warning_msg = "LLM returned empty or minimal structure. Using a generic placeholder structure."
                logger.warning(f"[{action_name}] {warning_msg}")
                if log_step_func: log_step_func(action_name, StepStatus.WARN.value, warning_msg, agent_name=self.name, task_id=task_id)
                structure_content = (
                    f"# Project Documentation\n\n## Overview\n\n## Features\n\n## Installation\n\n## Usage\n\n## Components / Modules\n\n"
                    f"*Note: Initial structure generation failed. This is a generic template.*\n\n"
                    f"**Analysis Used:**\n{analysis[:500]}...\n"
                )
                try:
                    await self.use_tool("update_task", task_id=task_id, details="Initial structure generation failed; attempting refinement with placeholder structure.", log_step_func=log_step_func)
                except Exception: pass

            # --- Refinement Step ---
            logger.info(f"[{action_name}] Starting refinement of the generated structure using react_loop.")
            refined_structure = await self.react_loop(
                initial_content=structure_content,
                revision_instruction=(
                    "**Refinement Goal:** Review and refine the provided Markdown document structure.\n"
                    "1.  Ensure a logical flow suitable for standard project documentation (e.g., Overview, Install, Usage, Components, Contributing).\n"
                    "2.  Use appropriate Markdown headers (`##` for main sections, `###` for sub-sections if necessary).\n"
                    "3.  Make sure headers are descriptive and relevant to typical project docs.\n"
                    "4.  Remove any conversational text, placeholders (unless intentionally added as a placeholder header), or generation artifacts.\n"
                    "5.  Ensure it starts with a single top-level `#` header."
                 ),
                task_id_to_update=task_id,
                refine_subtask_name="StructureRefinement",
                log_step_func=log_step_func, # Pass log func
                system_instruction=self.default_system_instruction, # Pass defaults
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                frequency_penalty=self.default_frequency_penalty,
                presence_penalty=self.default_presence_penalty
            )

            duration_structure_ms = int((time.monotonic() - start_time_structure) * 1000)
            logger.info(f"[{action_name}] Successfully completed structure generation and refinement for task (ID: {task_id})")
            if log_step_func: log_step_func(action_name, StepStatus.END.value, f"Successfully completed structure task {task_id}", agent_name=self.name, task_id=task_id, duration_ms=duration_structure_ms)
            # memory.current_task_id is cleared by successful react_loop
            return refined_structure

        except (asyncio.CancelledError, Exception) as e:
             duration_structure_ms = int((time.monotonic() - start_time_structure) * 1000)
             error_type = type(e).__name__
             log_level = logging.WARNING if isinstance(e, asyncio.CancelledError) else logging.ERROR
             status = StepStatus.CANCELLED if isinstance(e, asyncio.CancelledError) else StepStatus.ERROR
             logger.log(log_level, f"[{action_name}] Task failed or cancelled during structure generation/refinement for task {task_id}: {error_type}", exc_info=isinstance(e, Exception) and not isinstance(e, asyncio.CancelledError))
             if log_step_func: log_step_func(action_name, status.value, f"Structure task {task_id} {status.value.lower()}: {error_type}", agent_name=self.name, task_id=task_id, duration_ms=duration_structure_ms)
             # Clear memory on failure/cancellation if react_loop didn't handle it
             if self.memory.current_task_id == task_id:
                  self.memory.current_task_id = None
                  self.memory.last_updated = datetime.now(timezone.utc)
             raise


    async def enhance_design(self, task_id: str, structure: str, log_step_func=None) -> str:
        """
        Enhances the document structure by adding brief introductory sentences
        or placeholder text under each section header to create a draft.
        Uses `react_loop` for refinement and handles task updates. Logs steps.
        Updates agent's in-memory `current_task_id`.
        """
        action_name = f"{self.name} - EnhanceDesign"
        start_time_design = time.monotonic()
        if not await self._check_client_and_set_task_status(task_id, action_name, "Enhancing document design based on structure.", log_step_func):
             raise RuntimeError(f"Failed to start design enhancement task {task_id} due to missing client or task update error.")

        design_content = "" # Initialize design content variable
        try:
            # --- Initial Design Enhancement ---
            prompt = (
                f"**Role:** You are the {self.name}, focusing on fleshing out document structures.\n"
                f"**Task:** Enhance the provided Markdown document structure by adding brief introductory sentences or placeholder text under each section header.\n"
                f"**Input:** Markdown document structure (headers only).\n\n"
                f"**Instructions:**\n"
                f"1.  Take the input structure below.\n"
                f"2.  For each `##` and `###` section, add 1-2 sentences of introductory text or a clear placeholder (e.g., `[Details about installation...]`, `This section covers...`, `TODO: Describe usage examples.`).\n"
                f"3.  The goal is to make it look more like a readable document draft, indicating where detailed content is needed.\n"
                f"4.  Keep the original header structure intact.\n"
                f"5.  Respond *only* with the full, enhanced Markdown document content.\n\n"
                f"**Input Structure:**\n```markdown\n{structure}\n```\n\n"
                f"**Output:**\n"
            )
            logger.info(f"[{action_name}] Requesting initial design enhancement from LLM.")
            design_content = await self._execute_llm_call(
                prompt=prompt,
                task_description="Enhance Document Design (Initial)",
                log_step_func=log_step_func, # Pass log func
                action=f"{action_name}-Initial"
                # Use agent defaults
            )

            if not design_content or not design_content.strip() or design_content.strip() == structure.strip():
                 warning_msg = "LLM returned empty or unchanged design content. Using structure with basic placeholders as fallback."
                 logger.warning(f"[{action_name}] {warning_msg}")
                 if log_step_func: log_step_func(action_name, StepStatus.WARN.value, warning_msg, agent_name=self.name, task_id=task_id)
                 design_content_lines = structure.split('\n')
                 enhanced_lines = []
                 for line in design_content_lines:
                     enhanced_lines.append(line)
                     if line.strip().startswith("##"): enhanced_lines.append("\n[Details needed for this section...]\n")
                     elif line.strip().startswith("###"): enhanced_lines.append("\n[More specific details needed here...]\n")
                 design_content = "\n".join(enhanced_lines)
                 try:
                     await self.use_tool("update_task", task_id=task_id, details="Initial design enhancement failed; attempting refinement with placeholder content.", log_step_func=log_step_func)
                 except Exception: pass

            # --- Refinement Step ---
            logger.info(f"[{action_name}] Starting refinement of the enhanced design using react_loop.")
            refined_design = await self.react_loop(
                initial_content=design_content,
                revision_instruction=(
                    "**Refinement Goal:** Review and refine the enhanced Markdown document draft.\n"
                    "1.  Ensure the added placeholder text or introductory sentences are relevant to their respective section headers.\n"
                    "2.  Placeholders like `[Details needed]` should clearly indicate missing information.\n"
                    "3.  Improve overall clarity, readability, and Markdown formatting (e.g., consistent spacing).\n"
                    "4.  Remove any conversational text or generation artifacts.\n"
                    "5.  Maintain the original header structure."
                ),
                task_id_to_update=task_id,
                refine_subtask_name="DesignRefinement",
                log_step_func=log_step_func, # Pass log func
                system_instruction=self.default_system_instruction, # Pass defaults
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                frequency_penalty=self.default_frequency_penalty,
                presence_penalty=self.default_presence_penalty
            )

            duration_design_ms = int((time.monotonic() - start_time_design) * 1000)
            logger.info(f"[{action_name}] Successfully completed design enhancement and refinement for task (ID: {task_id})")
            if log_step_func: log_step_func(action_name, StepStatus.END.value, f"Successfully completed design task {task_id}", agent_name=self.name, task_id=task_id, duration_ms=duration_design_ms)
            # memory.current_task_id is cleared by successful react_loop
            return refined_design
        except (asyncio.CancelledError, Exception) as e:
             duration_design_ms = int((time.monotonic() - start_time_design) * 1000)
             error_type = type(e).__name__
             log_level = logging.WARNING if isinstance(e, asyncio.CancelledError) else logging.ERROR
             status = StepStatus.CANCELLED if isinstance(e, asyncio.CancelledError) else StepStatus.ERROR
             logger.log(log_level, f"[{action_name}] Task failed or cancelled during design enhancement/refinement for task {task_id}: {error_type}", exc_info=isinstance(e, Exception) and not isinstance(e, asyncio.CancelledError))
             if log_step_func: log_step_func(action_name, status.value, f"Design task {task_id} {status.value.lower()}: {error_type}", agent_name=self.name, task_id=task_id, duration_ms=duration_design_ms)
             # Clear memory on failure/cancellation if react_loop didn't handle it
             if self.memory.current_task_id == task_id:
                  self.memory.current_task_id = None
                  self.memory.last_updated = datetime.now(timezone.utc)
             raise


    async def add_knowledge_base(self, task_id: str, design: str, log_step_func=None) -> str:
        """
        Adds a 'Knowledge Base' or 'Technical Details' section to the document,
        populating it with inferred details or placeholders based on the existing content.
        Uses `react_loop` for refinement and handles task updates. Logs steps.
        Updates agent's in-memory `current_task_id`.
        """
        action_name = f"{self.name} - AddKnowledgeBase"
        start_time_kb = time.monotonic()
        if not await self._check_client_and_set_task_status(task_id, action_name, "Adding Knowledge Base / Technical Details section.", log_step_func):
             raise RuntimeError(f"Failed to start KB addition task {task_id} due to missing client or task update error.")

        kb_content = "" # Initialize KB content variable
        try:
            # --- Initial Knowledge Base Section Addition ---
            prompt = (
                f"**Role:** You are the {self.name}, enhancing technical documentation.\n"
                f"**Task:** Extend the provided Markdown document by adding a new section titled `## Knowledge Base / Technical Details` towards the end (e.g., before 'Contributing' or 'License' if they exist, otherwise near the end).\n"
                f"**Input:** Existing Markdown document draft.\n\n"
                f"**Instructions:**\n"
                f"1.  Review the entire input document below.\n"
                f"2.  Add a new section `## Knowledge Base / Technical Details`.\n"
                f"3.  Based *only* on the information *already present* in the document, populate this new section with:\n"
                f"    *   Any inferred technical assumptions made by the document.\n"
                f"    *   Potential limitations or edge cases hinted at.\n"
                f"    *   Areas explicitly marked as needing detail (e.g., `[Details needed]`).\n"
                f"    *   Complex concepts or components mentioned that might need deeper explanation later.\n"
                f"4.  If very little can be inferred, add relevant placeholders within this section (e.g., `- [Placeholder] Key algorithms used.`, `- [Placeholder] Dependencies and versions.`).\n"
                f"5.  Do *not* invent new technical details.\n"
                f"6.  Respond *only* with the full Markdown document, including the newly added section and its content.\n\n"
                f"**Input Document:**\n```markdown\n{design}\n```\n\n"
                f"**Output:**\n"
            )
            logger.info(f"[{action_name}] Requesting initial Knowledge Base section addition from LLM.")
            kb_content = await self._execute_llm_call(
                prompt=prompt,
                task_description="Add Knowledge Base Section (Initial)",
                log_step_func=log_step_func, # Pass log func
                action=f"{action_name}-Initial"
                # Use agent defaults
            )

            # --- Validation and Fallback for KB Section ---
            kb_header_present = "knowledge base" in kb_content.lower() or "technical details" in kb_content.lower()
            if not kb_content or not kb_content.strip() or not kb_header_present:
                 warning_msg = "LLM failed to add KB section or returned empty/invalid content. Appending a manual KB section placeholder."
                 logger.warning(f"[{action_name}] {warning_msg}")
                 if log_step_func: log_step_func(action_name, StepStatus.WARN.value, warning_msg, agent_name=self.name, task_id=task_id)
                 kb_content = design.strip() + (
                     f"\n\n## Knowledge Base / Technical Details\n\n"
                     f"*   [Placeholder] Initial population of this section failed or was incomplete.\n"
                     f"*   [Placeholder] Key technical assumptions or dependencies.\n"
                     f"*   [Placeholder] Potential limitations or areas for future work.\n"
                     f"*   Requires manual review based on the document content and project specifics."
                 )
                 try:
                     await self.use_tool("update_task", task_id=task_id, details="Initial KB section addition failed; attempting refinement with placeholder section.", log_step_func=log_step_func)
                 except Exception: pass

            # --- Refinement Step ---
            logger.info(f"[{action_name}] Starting refinement of the document with KB section using react_loop.")
            refined_kb_doc = await self.react_loop(
                initial_content=kb_content,
                revision_instruction=(
                    "**Refinement Goal:** Review and refine the entire Markdown document, with a focus on the 'Knowledge Base / Technical Details' section.\n"
                    "1.  Ensure the points listed in the KB section are relevant inferences drawn *from the rest of the document's content* or are clearly marked, valid placeholders for missing technical info.\n"
                    "2.  Improve the formatting, clarity, and integration of the KB section within the overall document.\n"
                    "3.  Check the entire document for consistency, readability, and correct Markdown.\n"
                    "4.  Remove any remaining conversational text or generation artifacts."
                ),
                task_id_to_update=task_id,
                refine_subtask_name="KBRefinement",
                log_step_func=log_step_func, # Pass log func
                system_instruction=self.default_system_instruction, # Pass defaults
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                frequency_penalty=self.default_frequency_penalty,
                presence_penalty=self.default_presence_penalty
            )

            duration_kb_ms = int((time.monotonic() - start_time_kb) * 1000)
            logger.info(f"[{action_name}] Successfully completed KB addition and refinement for task (ID: {task_id})")
            if log_step_func: log_step_func(action_name, StepStatus.END.value, f"Successfully completed KB task {task_id}", agent_name=self.name, task_id=task_id, duration_ms=duration_kb_ms)
            # memory.current_task_id is cleared by successful react_loop
            return refined_kb_doc
        except (asyncio.CancelledError, Exception) as e:
             duration_kb_ms = int((time.monotonic() - start_time_kb) * 1000)
             error_type = type(e).__name__
             log_level = logging.WARNING if isinstance(e, asyncio.CancelledError) else logging.ERROR
             status = StepStatus.CANCELLED if isinstance(e, asyncio.CancelledError) else StepStatus.ERROR
             logger.log(log_level, f"[{action_name}] Task failed or cancelled during KB addition/refinement for task {task_id}: {error_type}", exc_info=isinstance(e, Exception) and not isinstance(e, asyncio.CancelledError))
             if log_step_func: log_step_func(action_name, status.value, f"KB task {task_id} {status.value.lower()}: {error_type}", agent_name=self.name, task_id=task_id, duration_ms=duration_kb_ms)
             # Clear memory on failure/cancellation if react_loop didn't handle it
             if self.memory.current_task_id == task_id:
                  self.memory.current_task_id = None
                  self.memory.last_updated = datetime.now(timezone.utc)
             raise


# --- Dynamic Agent Loading ---
def load_agent_schema() -> List[Dict[str, Any]]:
    """
    Loads agent configuration definitions including name, role, goal, backstory, and tools.
    Placeholder: In a real application, this would load from a file (e.g., YAML/JSON).
    """
    # Schema updated to include goal, backstory, and tools per agent.
    # Tools are just example names for now.
    schema_json = '''
    [
        {
            "name": "Project Analyzer",
            "role": "Planner",
            "goal": "Analyze code structure and generate a high-level summary of the project's purpose and key components.",
            "backstory": "An experienced software architect AI specializing in understanding codebases quickly.",
            "tools": ["read_task", "read_all_tasks"]
        },
        {
            "name": "Doc Structurer",
            "role": "Crew",
            "goal": "Create a logical Markdown document structure (headers) based on the project analysis.",
            "backstory": "A technical writer AI focused on organizing information clearly.",
            "tools": ["create_task", "update_task", "read_task"]
        },
        {
            "name": "Content Enhancer",
            "role": "Crew",
            "goal": "Enhance the document structure by adding brief introductory text or placeholders under each section.",
            "backstory": "A documentation specialist AI that fleshes out outlines into readable drafts.",
            "tools": ["update_task", "read_task"]
        },
        {
            "name": "KB Compiler",
            "role": "Crew",
            "goal": "Add and populate a 'Knowledge Base / Technical Details' section based on information inferred from the draft.",
            "backstory": "An AI adept at extracting and summarizing technical details hidden within documentation drafts.",
            "tools": ["update_task", "read_task"]
        }
    ]
    '''
    try:
        schema = json.loads(schema_json)
        # Basic Validation
        if not isinstance(schema, list): raise ValueError("Agent schema must be a list of objects.")
        for i, item in enumerate(schema):
            if not isinstance(item, dict): raise ValueError(f"Schema item at index {i} must be an object.")
            if not item.get("name") or not isinstance(item["name"], str):
                raise ValueError(f"Schema item at index {i} must have a non-empty 'name' (string).")
            if not item.get("role") or not isinstance(item["role"], str):
                raise ValueError(f"Schema item at index {i} must have a non-empty 'role' (string).")
            # Goal and backstory are optional but should be strings if present
            if "goal" in item and not isinstance(item["goal"], str):
                 raise ValueError(f"Schema item '{item['name']}' has non-string 'goal'.")
            if "backstory" in item and not isinstance(item["backstory"], str):
                 raise ValueError(f"Schema item '{item['name']}' has non-string 'backstory'.")
            # Tools are optional but should be list of strings if present
            if "tools" in item and not isinstance(item["tools"], list):
                raise ValueError(f"Schema item '{item['name']}' has non-list 'tools'.")
            if "tools" in item and not all(isinstance(t, str) for t in item["tools"]):
                 raise ValueError(f"Schema item '{item['name']}' has non-string elements in 'tools'.")

        logger.info(f"Loaded agent schema definition with {len(schema)} agents.")
        return schema
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load or validate agent schema: {e}. Using fallback default agents.")
        # Return a minimal fallback schema if loading/validation fails
        return [
            {"name": "Fallback Planner", "role": "Planner", "goal": "Perform basic analysis", "backstory": "Fallback agent", "tools": []},
            {"name": "Fallback Crew", "role": "Crew", "goal": "Perform basic content generation", "backstory": "Fallback agent", "tools": []},
        ]

def create_dynamic_agents(client: Optional[GeminiModelClient], tools: List[ToolSchema]) -> List[BaseAgent]:
    """
    Creates agent instances based on the loaded schema definition.
    Assigns the appropriate agent class (PlannerAgent, CrewAgent) based on the 'role'.
    Injects the LLM client (if available), tools, goal, and backstory into each agent.
    Includes fallback mechanisms if essential roles are missing.
    """
    agent_schema = load_agent_schema() # Load the definitions
    agents: List[BaseAgent] = []
    # Map available tools by name for easy lookup
    available_tools_map: Dict[str, ToolSchema] = {tool.name: tool for tool in tools} if tools else {}

    agent_classes = {
        "planner": PlannerAgent,
        "crew": CrewAgent,
        "base": BaseAgent # Fallback for unknown roles
    }

    logger.info(f"Creating agents based on schema (LLM Client available: {client is not None})...")
    for config in agent_schema:
        role = config.get("role", "Base").strip().lower() # Normalize role to lowercase
        name = config.get("name", f"Agent Role ({role.capitalize()})") # Use defined name or generate default
        goal = config.get("goal") # Get optional goal
        backstory = config.get("backstory") # Get optional backstory
        # Get the list of tool *names* requested for this agent from the schema
        requested_tool_names = config.get("tools", [])
        # Filter the globally available tools to get only the ones requested for this agent
        agent_specific_tools = [available_tools_map[tool_name] for tool_name in requested_tool_names if tool_name in available_tools_map]

        if not agent_specific_tools and requested_tool_names:
             logger.warning(f"Agent '{name}' requested tools {requested_tool_names}, but some/all were not found in the available tools map: {list(available_tools_map.keys())}")

        agent_class = agent_classes.get(role, BaseAgent)
        logger.info(f"Attempting to create agent: Name='{name}', Role='{role}', Class={agent_class.__name__}, Goal='{goal}', Backstory='{(backstory or '')[:30]}...'")

        try:
            # Pass goal and backstory to the agent constructor
            agent_instance = agent_class(client, agent_specific_tools, name=name, goal=goal, backstory=backstory)
            agents.append(agent_instance)
            logger.info(f"Successfully created agent: '{name}' ({agent_class.__name__}) with {len(agent_specific_tools)} tools: {[t.name for t in agent_specific_tools]}")
        except Exception as e:
            logger.error(f"Failed to create agent '{name}' (Role: '{role}', Class: {agent_class.__name__}): {e}", exc_info=True)

    # --- Ensure Essential Roles are Present (Fallback Mechanism) ---
    has_planner = any(isinstance(a, PlannerAgent) for a in agents)
    has_crew = any(isinstance(a, CrewAgent) for a in agents)

    if client: # Only add fallbacks if LLM is available
        if not has_planner:
            logger.warning("[create_dynamic_agents] No agent with 'Planner' role created from schema. Adding a fallback PlannerAgent.")
            try:
                 # Provide all tools to fallback agents? Or minimal set? Let's give all.
                 agents.insert(0, PlannerAgent(client, tools, name="Fallback Planner", goal="Perform analysis", backstory="Fallback"))
            except Exception as e:
                 logger.error(f"Failed to create Fallback Planner agent: {e}", exc_info=True)

        if not has_crew:
            logger.warning("[create_dynamic_agents] No agent with 'Crew' role created from schema. Adding a fallback CrewAgent.")
            planner_index = next((i for i, a in enumerate(agents) if isinstance(a, PlannerAgent)), -1)
            insert_index = planner_index + 1 if planner_index != -1 else len(agents)
            try:
                 agents.insert(insert_index, CrewAgent(client, tools, name="Fallback Crew", goal="Generate content", backstory="Fallback"))
            except Exception as e:
                 logger.error(f"Failed to create Fallback Crew agent: {e}", exc_info=True)

    elif not agents:
         logger.error("[create_dynamic_agents] CRITICAL: No agents could be created (schema failed/empty?) and no LLM client is available. Orchestration will fail.")

    logger.info(f"Agent creation complete. Final active agents ({len(agents)}): {[f'{a.name} ({a.role})' for a in agents]}")
    return agents


# --- Manager Class (Orchestrator) ---
class Manager:
    """
    Orchestrates the entire documentation generation workflow.
    - Initializes and manages agents based on roles.
    - Manages the overall process flow (analyze -> structure -> design -> KB -> polish).
    - Creates tasks using the TaskManager and assigns them to appropriate agents.
    - Waits for task completion and handles dependencies between steps.
    - Logs orchestration steps to console and StepManager.
    - Uses its own LLM client for final polishing if available.
    - Saves final output and metadata according to the defined schema.
    """
    def __init__(self, client: Optional[GeminiModelClient], task_manager: TaskManager, step_manager: StepManager): # Add step_manager
        if client is not None and not isinstance(client, GeminiClient):
             logger.error(f"[Manager] CRITICAL: Received incorrect client type: {type(client)}. Expected custom GeminiClient or None.")
             raise TypeError("Manager requires an instance of the custom GeminiClient or None.")
        self.client = client # Store the client instance (can be None)
        self.task_manager = task_manager
        self.step_manager = step_manager # Initialize StepManager
        self.global_tools = task_manager.get_tools() # Get all available tools from task manager
        # Create agent instances using the dynamic loading function, passing all tools
        self.agents = create_dynamic_agents(client, self.global_tools)
        self.name = "Manager Orchestrator" # Name for the orchestrator itself
        # Generate a unique ID for this specific orchestration run for tracking/logging
        self.orchestration_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:-3]}"
        # Define the specific output directory for this run
        self.output_dir = OUTPUT_BASE_DIR / self.orchestration_id
        self.step_counter = 0 # Counter for logging steps
        self.tasks_created_this_run: List[str] = [] # Track tasks created by this specific run
        self.start_time = time.monotonic() # Record start time for duration calculation

        # Log manager initialization details
        client_status = "with custom GeminiClient" if self.client else "WITHOUT LLM client"
        logger.info(f"[{self.name}] Initialized ({client_status}). Orchestration ID: {self.orchestration_id}")
        logger.info(f"[{self.name}] Output directory for this run: {self.output_dir}")
        # Ensure output directory exists *early*
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"[{self.name}] CRITICAL: Failed to create output directory {self.output_dir}: {e}. Orchestration may fail to save results.")

        # Initial log step for orchestration start
        self._log_step("Manager Initialization", StepStatus.INFO.value, f"Manager initialized ({client_status}). Output Dir: {self.output_dir.relative_to(BASE_DIR)}", agent_name=self.name)
        if not self.agents:
            logger.warning(f"[{self.name}] Initialization completed with NO AGENTS. Orchestration will likely fail.")
            self._log_step("Manager Initialization", StepStatus.WARN.value, "No agents created during initialization.", agent_name=self.name)
        elif not self.client:
             logger.warning(f"[{self.name}] Initialized without LLM client. Agent LLM calls and final polish will fail.")
             self._log_step("Manager Initialization", StepStatus.WARN.value, "Manager initialized without LLM client.", agent_name=self.name)

    # --- Step Logging Method ---
    def _log_step(self, step_name: str, step_status: str, message: str = "", task_id: Optional[str] = None, agent_name: Optional[str] = None, duration_ms: Optional[int] = None):
        """
        Logs orchestration steps to both the console logger and the shared StepManager log file.
        """
        self.step_counter += 1
        log_level = logging.INFO # Default
        status_enum = StepStatus.INFO # Default enum
        try:
             status_enum = StepStatus(step_status) # Validate/convert string to enum
        except ValueError:
             logger.warning(f"[Manager Log Step] Invalid step status '{step_status}' received. Defaulting to INFO.")
             status_enum = StepStatus.INFO

        status_upper = status_enum.value.strip().upper()
        if status_upper in ["ERROR", "FAIL", "TIMEOUT", "BLOCKED", "CRITICAL"]: log_level = logging.ERROR
        elif status_upper in ["WARN", "WARNING", "CANCELLED", "SKIP"]: log_level = logging.WARNING

        console_log_message = f"[{self.orchestration_id} | MGR_STEP {self.step_counter:03d}] [{step_name}] [{status_enum.value}] {message}"
        if duration_ms is not None:
             console_log_message += f" (Duration: {duration_ms}ms)"
        logger.log(log_level, console_log_message)

        step_log_entry = StepSchema(
            orchestration_id=self.orchestration_id, # Tag with the current run ID
            task_id=task_id,
            agent_name=agent_name or self.name,
            step_name=step_name,
            status=status_enum,
            details=message,
            duration_ms=duration_ms,
        )
        # Asynchronously save the step log using the StepManager (appends to shared file)
        asyncio.create_task(self.step_manager.create_step_log(step_log_entry))

    # --- Wait for Task Method ---
    async def _wait_for_task(self, task_id: str, timeout: float = 180.0) -> TaskSchema:
        """
        Polls the TaskManager for the status of a specific task ID until it reaches
        a terminal state (DONE, ERROR, CANCELLED) or the specified timeout is reached.
        Handles task not found errors and timeout scenarios, logging steps appropriately.
        """
        wait_action_name = f"{self.name} - WaitTask"
        start_time = time.monotonic()
        self._log_step(wait_action_name, StepStatus.START.value, f"Waiting for task '{task_id}' to complete (Timeout: {timeout:.1f}s)", task_id=task_id, agent_name=self.name)

        try:
            while True:
                elapsed = time.monotonic() - start_time
                elapsed_ms = int(elapsed * 1000)

                # Check for timeout
                if elapsed >= timeout:
                    timeout_msg = f"Task '{task_id}' timed out after {elapsed:.2f}s waiting for completion."
                    self._log_step(wait_action_name, StepStatus.ERROR.value, timeout_msg, task_id=task_id, agent_name=self.name, duration_ms=elapsed_ms)
                    timed_out_task = await self.task_manager._read_task_impl(task_id) # Check cache first
                    if not timed_out_task: # If not in cache, maybe it finished but cache didn't update? Reload cache.
                         logger.warning(f"[{wait_action_name}] Task {task_id} not in cache on timeout. Reloading task cache from file.")
                         await self.task_manager.initialize() # Re-initialize to load latest from file
                         timed_out_task = await self.task_manager._read_task_impl(task_id)

                    if timed_out_task:
                         if timed_out_task.status in [TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED]:
                             self._log_step(wait_action_name, StepStatus.WARN.value, f"Task '{task_id}' finished ({timed_out_task.status.value}) concurrently with timeout check.", task_id=task_id, agent_name=self.name)
                             return timed_out_task
                         else:
                             self._log_step(wait_action_name, StepStatus.INFO.value, f"Marking task '{task_id}' as ERROR due to timeout.", task_id=task_id, agent_name=self.name)
                             error_details = f"Task timed out after {timeout:.1f}s. Last known status: {timed_out_task.status.value}."
                             updated_task = await self.task_manager._update_task_impl(task_id, status=TaskStatus.ERROR, details=error_details)
                             return updated_task or timed_out_task # Return updated task if successful, otherwise the timed out one
                    else:
                         timeout_error_msg = f"Task '{task_id}' timed out ({timeout:.1f}s) and could not be found even after reload."
                         self._log_step(wait_action_name, StepStatus.ERROR.value, timeout_error_msg, task_id=task_id, agent_name=self.name)
                         # Return a dummy error task schema
                         return TaskSchema( id=task_id, name="<Timed Out & Not Found>", status=TaskStatus.ERROR, details=timeout_error_msg, orchestration_id=self.orchestration_id )

                # Read task status from the cache
                task = await self.task_manager._read_task_impl(task_id)

                # If task not in cache, maybe it was just created by another process? Reload cache.
                # This adds overhead but increases robustness with shared file.
                if not task:
                     logger.debug(f"[{wait_action_name}] Task {task_id} not found in cache, reloading cache.")
                     await self.task_manager.initialize() # Reload cache from file
                     task = await self.task_manager._read_task_impl(task_id)
                     if not task:
                         # Still not found after reload, likely an issue or deleted.
                         error_msg = f"Task '{task_id}' could not be found or was deleted while waiting (after {elapsed:.2f}s, cache reloaded)."
                         self._log_step(wait_action_name, StepStatus.ERROR.value, error_msg, task_id=task_id, agent_name=self.name, duration_ms=elapsed_ms)
                         return TaskSchema( id=task_id, name="<Deleted Task>", status=TaskStatus.ERROR, details=error_msg, orchestration_id=self.orchestration_id )

                # Check terminal state
                if task.status in [TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED]:
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    self._log_step(wait_action_name, StepStatus.END.value, f"Task '{task_id}' reached terminal state: {task.status.value}", task_id=task_id, agent_name=self.name, duration_ms=duration_ms)
                    return task

                await asyncio.sleep(1.5) # Poll interval

        except Exception as e:
             duration_ms = int((time.monotonic() - start_time) * 1000)
             error_type = type(e).__name__
             error_msg = f"Unexpected error while waiting for task '{task_id}' after {duration_ms}ms: {error_type}: {e}"
             self._log_step(wait_action_name, StepStatus.ERROR.value, error_msg, task_id=task_id, agent_name=self.name, duration_ms=duration_ms)
             logger.exception(f"[{wait_action_name}] Unexpected error waiting for task {task_id}.")
             # Attempt to mark the task as ERROR in the shared log
             try:
                 # Ensure task exists before updating
                 task_to_update = await self.task_manager._read_task_impl(task_id)
                 if task_to_update:
                     await self.task_manager._update_task_impl(task_id, status=TaskStatus.ERROR, details=f"Critical error during wait loop: {error_type}")
             except Exception as update_err:
                 logger.error(f"[{wait_action_name}] Failed to mark task '{task_id}' as error after wait failure: {update_err}")
             # Return a dummy error task schema
             return TaskSchema( id=task_id, name="<Task Wait Critical Error>", status=TaskStatus.ERROR, details=error_msg, orchestration_id=self.orchestration_id )

    def _clean_final_markdown(self, markdown_text: str) -> str:
        """Helper to perform basic cleaning on the final generated Markdown."""
        if not markdown_text or not isinstance(markdown_text, str):
             return ""

        patterns_to_remove = [
            re.compile(r"^```markdown\s*\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"\n```\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^As requested.*?\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Here is the refined.*?\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Here's the improved.*?\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^\*\*Output:\*\*\s*\n?", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Okay, here is the.*?\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"\[Details needed.*?\]", re.IGNORECASE),
            re.compile(r"\[Placeholder.*?\]", re.IGNORECASE),
            re.compile(r"TODO:.*", re.IGNORECASE),
            re.compile(r"^\s*Note:.*generation failed.*?\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^\s*\*Note:.*generic template\.\*\s*\n?", re.IGNORECASE | re.MULTILINE),
            # Be careful with removing Analysis Used, might be legitimate content
            # re.compile(r"^\*\*Analysis Used:\*\*.*", re.IGNORECASE | re.DOTALL),
            re.compile(r"\n{3,}", re.MULTILINE), # Replace 3+ newlines with 2
        ]

        cleaned_text = markdown_text
        for pattern in patterns_to_remove:
            cleaned_text = pattern.sub("", cleaned_text)

        return cleaned_text.strip()


    async def _create_and_assign_task(self, name: str, details: str, agent: BaseAgent) -> TaskSchema:
        """Helper to create a task and log the assignment."""
        task_action_name = f"{self.name} - CreateTask-{name.replace(' ', '')}"
        self._log_step(task_action_name, StepStatus.START.value, f"Creating task '{name}' and assigning to Agent '{agent.name}'...", agent_name=self.name)
        task = await self.task_manager._create_task_impl(
            name=name,
            details=details,
            assigned_to=agent.name,
            orchestration_id=self.orchestration_id # Link task to this run
        )
        self.tasks_created_this_run.append(task.id) # Track created task ID for this run
        self._log_step(task_action_name, StepStatus.ASSIGN.value, f"Task '{task.id}' created and assigned to '{agent.name}'.", task_id=task.id, agent_name=self.name)
        return task

    async def _gather_statistics(self, end_time: float) -> StatisticsSchema:
        """Gathers statistics about the completed orchestration run."""
        total_execution_time = end_time - self.start_time
        # Read only tasks associated with *this* orchestration run
        tasks = await self.task_manager._read_all_tasks_impl(orchestration_id_filter=self.orchestration_id)
        total_tasks = len(tasks)
        successful_tasks = len([t for t in tasks if t.status == TaskStatus.DONE])
        failed_tasks = len([t for t in tasks if t.status in [TaskStatus.ERROR, TaskStatus.CANCELLED]])

        # --- Placeholder/Not Implemented Stats ---
        total_iterations = None # Not explicitly tracked per orchestration
        retried_tasks = None # Task retry logic not implemented
        max_parallel_tasks = None # Parallel execution not explicitly managed/tracked
        # --- Calculated Stats ---
        avg_task_duration = None
        longest_task_duration = None
        shortest_task_duration = None

        completed_tasks = [t for t in tasks if t.status == TaskStatus.DONE]
        durations = []
        if completed_tasks:
            for task in completed_tasks:
                # Ensure timestamps are valid datetime objects
                created_at_dt = task.created_at if isinstance(task.created_at, datetime) else None
                updated_at_dt = task.updated_at if isinstance(task.updated_at, datetime) else None

                if updated_at_dt and created_at_dt:
                    duration = (updated_at_dt - created_at_dt).total_seconds()
                    # Basic sanity check for duration (e.g., ignore negative durations)
                    if duration >= 0:
                        durations.append(duration)
                    else:
                         logger.warning(f"[Stats] Task {task.id} has negative duration ({duration}s). Skipping.")

            if durations:
                 avg_task_duration = sum(durations) / len(durations)
                 longest_task_duration = max(durations)
                 shortest_task_duration = min(durations)

        # Count steps for *this* orchestration run from the StepManager
        # This requires reading the step log file again, filtering by ID.
        run_steps = await self.step_manager.get_steps(orchestration_id=self.orchestration_id, limit=5000) # Use a high limit
        total_steps_this_run = len(run_steps)

        return StatisticsSchema(
            total_steps=total_steps_this_run, # Use count from filtered steps
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            total_execution_time=round(total_execution_time, 3),
            total_iterations=total_iterations, # Placeholder
            retried_tasks=retried_tasks, # Placeholder
            max_parallel_tasks=max_parallel_tasks, # Placeholder
            avg_task_duration=round(avg_task_duration, 3) if avg_task_duration is not None else None,
            longest_task_duration=round(longest_task_duration, 3) if longest_task_duration is not None else None,
            shortest_task_duration=round(shortest_task_duration, 3) if shortest_task_duration is not None else None,
        )

    async def _save_output_metadata(self, final_result_file_name: str, stats: StatisticsSchema):
        """Saves the OrchestrationOutputSchema metadata to a JSON file within the run's output directory."""
        # Output directory is already created in __init__
        metadata_file = self.output_dir / "orchestration_output.json"

        # Get relative paths for *shared* log files from BASE_DIR
        try:
            step_log_rel = step_manager_instance.file_path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            step_log_rel = str(step_manager_instance.file_path) # Use absolute if not relative

        try:
            task_log_rel = task_manager_instance.file_path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            task_log_rel = str(task_manager_instance.file_path)

        memory_log_rel = None # Memory logs not saved to file currently
        # if MEMORY_LOG_FILE.exists(): # Example if memory log file was used
        #     try: memory_log_rel = MEMORY_LOG_FILE.relative_to(BASE_DIR).as_posix()
        #     except ValueError: memory_log_rel = str(MEMORY_LOG_FILE)

        # Determine final result file type
        final_result_type = [Path(final_result_file_name).suffix[1:]] if Path(final_result_file_name).suffix else ["unknown"]

        # Create AgentItem list including goal and backstory
        agent_list_items = []
        for a in self.agents:
             agent_list_items.append(AgentItem(
                 agent_name=a.name,
                 role=a.role,
                 goal=a.goal, # Read from agent instance
                 backstory=a.backstory, # Read from agent instance
                 tools=list(a.tools.keys()) # Get tool names available to this agent
             ))

        output_schema_data = OrchestrationOutputSchema(
            dir=self.output_dir.relative_to(BASE_DIR).as_posix(), # Dir relative to project root
            orchestration_name="Automated Documentation Generation", # Example name
            orchestration_goal="Generate project documentation based on code analysis.", # Example goal
            orchestration_instruction="Analyze files -> Generate analysis -> Create structure -> Enhance design -> Add KB -> Polish -> Save result.", # Example instruction
            step_log=step_log_rel, # Path to shared step log relative to BASE_DIR
            task_log=task_log_rel, # Path to shared task log relative to BASE_DIR
            memory_log=memory_log_rel, # Path to shared memory log (if any)
            agent_list=agent_list_items, # Use the populated list
            statistics=stats,
            final_result_file_type=final_result_type,
            final_result=final_result_file_name # Just the filename, relative to 'dir'
        )

        try:
            async with aiofiles.open(metadata_file, "w", encoding="utf-8") as f:
                await f.write(output_schema_data.model_dump_json(indent=2))
            self._log_step("Save Output Metadata", StepStatus.INFO.value, f"Saved orchestration metadata to {metadata_file.relative_to(BASE_DIR)}", agent_name=self.name)
        except Exception as e:
            self._log_step("Save Output Metadata", StepStatus.ERROR.value, f"Failed to save metadata: {e}", agent_name=self.name)


    # --- Orchestration Method ---
    async def orchestrate(self, folder: str) -> str:
        """
        Runs the main orchestration logic, logging steps via StepManager.
        Saves final markdown and metadata. Returns the final markdown content.
        """
        main_action_name = f"{self.name} - Orchestrate"
        self._log_step(main_action_name, StepStatus.START.value, f"Starting orchestration run. Target folder: '{folder}'", agent_name=self.name)

        # Default final output in case of early error
        final_result_file_name = "result.md" # Define standard output filename
        final_md_file_path = self.output_dir / final_result_file_name
        final_markdown = f"# Orchestration Error\n\nOrchestration ID: {self.orchestration_id}\nAn unexpected error occurred before generation could complete. Check logs and metadata file (if created)."
        orchestration_success = False # Flag to track success

        # --- Initial Checks ---
        if not self.client:
            error_msg = "Orchestration blocked: LLM client (custom GeminiClient) is not available or failed to initialize. Check configuration (API key, imports) and logs."
            self._log_step(main_action_name, StepStatus.BLOCK.value, error_msg, agent_name=self.name)
            if not GeminiClient: error_msg += "\nReason: Custom GeminiClient class ('app.llm.gemini.GeminiClient') could not be imported."
            elif not GEMINI_API_KEY: error_msg += "\nReason: `GEMINI_API_KEY` environment variable not set."
            else: error_msg += "\nReason: Check `app.llm.gemini.GeminiClient` instantiation or connectivity."
            # Save metadata even on early failure
            stats = await self._gather_statistics(time.monotonic())
            await self._save_output_metadata(final_result_file_name="<blocked>", stats=stats)
            return f"# Configuration Error\n\nOrchestration ID: {self.orchestration_id}\n{error_msg}"

        if not self.agents:
             error_msg = "Orchestration blocked: No agents were successfully created (check agent schema loading). Cannot proceed."
             self._log_step(main_action_name, StepStatus.BLOCK.value, error_msg, agent_name=self.name)
             stats = await self._gather_statistics(time.monotonic())
             await self._save_output_metadata(final_result_file_name="<blocked>", stats=stats)
             return f"# Agent Initialization Error\n\nOrchestration ID: {self.orchestration_id}\n{error_msg}"

        # --- Agent Role Assignment ---
        try:
            planner = next((a for a in self.agents if isinstance(a, PlannerAgent)), None)
            crew_agents = [a for a in self.agents if isinstance(a, CrewAgent)]
            if not planner: raise ValueError("Required 'Planner' role agent is missing.")
            if not crew_agents: raise ValueError("Required 'Crew' role agent(s) are missing.")

            # Assign agents based on schema order or specific names if needed
            # For simplicity, use the order they appear in the loaded schema/list
            structure_agent = crew_agents[0]
            design_agent = crew_agents[1] if len(crew_agents) > 1 else crew_agents[0] # Reuse if only one
            kb_agent = crew_agents[2] if len(crew_agents) > 2 else crew_agents[-1] # Use third or last

            self._log_step(main_action_name, StepStatus.INFO.value, f"Agents assigned: Planner='{planner.name}', Structure='{structure_agent.name}', Design='{design_agent.name}', KB='{kb_agent.name}'", agent_name=self.name)
        except (ValueError, IndexError) as e:
            error_msg = f"Agent assignment failed: {e}. Ensure schema defines enough agents for required roles (Planner, Crew)."
            self._log_step(main_action_name, StepStatus.ERROR.value, error_msg, agent_name=self.name)
            stats = await self._gather_statistics(time.monotonic())
            await self._save_output_metadata(final_result_file_name="<error>", stats=stats)
            return f"# Agent Role Error\n\nOrchestration ID: {self.orchestration_id}\n{error_msg}"
        except Exception as e:
             self._log_step(main_action_name, StepStatus.ERROR.value, f"Unexpected error during agent assignment: {type(e).__name__}: {e}", agent_name=self.name)
             stats = await self._gather_statistics(time.monotonic())
             await self._save_output_metadata(final_result_file_name="<error>", stats=stats)
             return f"# Agent Assignment Error\n\nOrchestration ID: {self.orchestration_id}\nUnexpected error: {e}"

        # --- Orchestration Workflow Steps ---
        try:
            # === Step 1: Analyze Project Files (System Task) ===
            step1_name = "Analyze Project Files"
            self._log_step(step1_name, StepStatus.START.value, f"Analyzing project files in '{folder}'...", agent_name="System")
            start_time_step1 = time.monotonic()
            files_info = await analyze_files(folder, self._log_step) # Pass log_step
            duration_step1_ms = int((time.monotonic() - start_time_step1) * 1000)
            if not files_info:
                self._log_step(step1_name, StepStatus.WARN.value, "File analysis found no suitable Python files.", agent_name="System", duration_ms=duration_step1_ms)
            else:
                self._log_step(step1_name, StepStatus.END.value, f"File analysis complete. Found info for {len(files_info)} files.", agent_name="System", duration_ms=duration_step1_ms)


            # === Step 2: Generate Project Analysis (Planner Agent) ===
            step2_name = "Generate Project Analysis"
            analysis_task = await self._create_and_assign_task(
                name=step2_name,
                details=f"Analyze file info for folder: {os.path.basename(folder)}",
                agent=planner
            )
            analysis_future = asyncio.create_task(planner.generate_analysis(analysis_task.id, files_info, self._log_step)) # Pass log_step
            analysis_task_result = await self._wait_for_task(analysis_task.id, timeout=300.0) # Wait with logging

            if analysis_task_result.status != TaskStatus.DONE:
                error_details = analysis_task_result.details or f"Status was {analysis_task_result.status.value}"
                raise ValueError(f"Analysis task ('{analysis_task.id}') failed. Details: {error_details}")
            analysis_content = analysis_task_result.result
            if not analysis_content or not isinstance(analysis_content, str):
                analysis_content = analysis_task_result.details if isinstance(analysis_task_result.details, str) else f"Analysis task {analysis_task.id} finished DONE but had no valid string result."
                self._log_step(step2_name, StepStatus.WARN.value, f"Task '{analysis_task.id}' completed but result was invalid/empty. Using details as fallback.", task_id=analysis_task.id, agent_name=self.name)
            self._log_step(step2_name, StepStatus.COMPLETE.value, f"Task '{analysis_task.id}' completed by '{planner.name}'. Result length: {len(analysis_content)}", task_id=analysis_task.id, agent_name=self.name)


            # === Step 3: Generate Document Structure (Crew Agent 1) ===
            step3_name = "Generate Document Structure"
            structure_task = await self._create_and_assign_task(
                name=step3_name,
                details=f"Input: Analysis from task '{analysis_task.id}'",
                agent=structure_agent
            )
            structure_future = asyncio.create_task(structure_agent.generate_structure(structure_task.id, analysis_content, self._log_step)) # Pass log_step
            structure_task_result = await self._wait_for_task(structure_task.id, timeout=240.0)

            if structure_task_result.status != TaskStatus.DONE:
                error_details = structure_task_result.details or f"Status was {structure_task_result.status.value}"
                raise ValueError(f"Structure task ('{structure_task.id}') failed. Details: {error_details}")
            structure_content = structure_task_result.result
            if not structure_content or not isinstance(structure_content, str):
                structure_content = structure_task_result.details if isinstance(structure_task_result.details, str) else f"Structure task {structure_task.id} finished DONE but had no valid string result."
                self._log_step(step3_name, StepStatus.WARN.value, f"Task '{structure_task.id}' completed but result was invalid/empty. Using details as fallback.", task_id=structure_task.id, agent_name=self.name)
            self._log_step(step3_name, StepStatus.COMPLETE.value, f"Task '{structure_task.id}' completed by '{structure_agent.name}'. Result length: {len(structure_content)}", task_id=structure_task.id, agent_name=self.name)


            # === Step 4: Enhance Document Design (Crew Agent 2) ===
            step4_name = "Enhance Document Design"
            design_task = await self._create_and_assign_task(
                name=step4_name,
                details=f"Input: Structure from task '{structure_task.id}'",
                agent=design_agent
            )
            design_future = asyncio.create_task(design_agent.enhance_design(design_task.id, structure_content, self._log_step)) # Pass log_step
            design_task_result = await self._wait_for_task(design_task.id, timeout=240.0)

            if design_task_result.status != TaskStatus.DONE:
                error_details = design_task_result.details or f"Status was {design_task_result.status.value}"
                raise ValueError(f"Design enhancement task ('{design_task.id}') failed. Details: {error_details}")
            design_content = design_task_result.result
            if not design_content or not isinstance(design_content, str):
                design_content = design_task_result.details if isinstance(design_task_result.details, str) else f"Design task {design_task.id} finished DONE but had no valid string result."
                self._log_step(step4_name, StepStatus.WARN.value, f"Task '{design_task.id}' completed but result was invalid/empty. Using details as fallback.", task_id=design_task.id, agent_name=self.name)
            self._log_step(step4_name, StepStatus.COMPLETE.value, f"Task '{design_task.id}' completed by '{design_agent.name}'. Result length: {len(design_content)}", task_id=design_task.id, agent_name=self.name)


            # === Step 5: Add Knowledge Base Section (Crew Agent 3) ===
            step5_name = "Add Knowledge Base Section"
            kb_task = await self._create_and_assign_task(
                name=step5_name,
                details=f"Input: Enhanced design from task '{design_task.id}'",
                agent=kb_agent
            )
            kb_future = asyncio.create_task(kb_agent.add_knowledge_base(kb_task.id, design_content, self._log_step)) # Pass log_step
            kb_task_result = await self._wait_for_task(kb_task.id, timeout=240.0)

            if kb_task_result.status != TaskStatus.DONE:
                error_details = kb_task_result.details or f"Status was {kb_task_result.status.value}"
                raise ValueError(f"KB task ('{kb_task.id}') failed. Details: {error_details}")
            kb_doc_content = kb_task_result.result
            if not kb_doc_content or not isinstance(kb_doc_content, str):
                 kb_doc_content = kb_task_result.details if isinstance(kb_task_result.details, str) else f"KB task {kb_task.id} finished DONE but had no valid string result."
                 self._log_step(step5_name, StepStatus.WARN.value, f"Task '{kb_task.id}' completed but result was invalid/empty. Using details as fallback.", task_id=kb_task.id, agent_name=self.name)
            self._log_step(step5_name, StepStatus.COMPLETE.value, f"Task '{kb_task.id}' completed by '{kb_agent.name}'. Result length: {len(kb_doc_content)}", task_id=kb_task.id, agent_name=self.name)


            # === Step 6: Final Polish (using Manager's client via a temporary BaseAgent) ===
            step6_name = "Final Document Polish"
            self._log_step(step6_name, StepStatus.START.value, "Performing final polish on the generated document...", agent_name=self.name)
            start_time_step6 = time.monotonic()
            release_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            final_doc_raw = kb_doc_content # Start with KB doc as input for polish

            if self.client:
                try:
                    # Use a temporary agent instance for the final polish LLM call
                    temp_polisher = BaseAgent(self.client, tools=None, name="Manager-Polisher", role="Polisher", goal="Polish final document", backstory="Final editor")
                    polish_prompt = (
                         f"**Role:** You are a final editor.\n"
                         f"**Task:** Perform a final polish on the Markdown document below. Your goal is to make it clean, consistent, and ready for use as project documentation (e.g., README).\n"
                         f"**Input:** Markdown document draft.\n\n"
                         f"**Instructions:**\n"
                         f"1.  Review the entire document for consistent Markdown formatting (headers, lists, code blocks, spacing).\n"
                         f"2.  Ensure a professional tone and clear language.\n"
                         f"3.  Check for and remove any remaining obvious placeholders (like `[Details needed]`, `[Placeholder]`, `TODO:`), unless they represent actual code comments or similar.\n"
                         f"4.  Remove any conversational filler text, self-corrections, or explicit instructions from previous generation steps.\n"
                         f"5.  Add a generation timestamp comment at the very top like: `<!-- Generated on: {release_date_str} -->`\n"
                         f"6.  Respond *only* with the clean, polished Markdown document content. Do not add any introduction or explanation.\n\n"
                         f"**Document to Polish:**\n```markdown\n{kb_doc_content}\n```\n\n"
                         f"**Polished Output:**\n"
                     )
                    final_doc_raw = await temp_polisher._execute_llm_call( prompt=polish_prompt, task_description="Final Polish", log_step_func=self._log_step, action="FinalPolish", temperature=0.2 ) # Pass log_step
                except Exception as polish_err:
                    error_type = type(polish_err).__name__
                    self._log_step(step6_name, StepStatus.WARN.value, f"Final polish step failed: {error_type}: {polish_err}. Using pre-polish content.", agent_name=self.name)
                    # Keep final_doc_raw as kb_doc_content (pre-polish)
            else:
                 self._log_step(step6_name, StepStatus.SKIP.value, "Skipping final polish step as Manager has no LLM client.", agent_name=self.name)
                 # Keep final_doc_raw as kb_doc_content

            duration_step6_ms = int((time.monotonic() - start_time_step6) * 1000)
            self._log_step(step6_name, StepStatus.END.value, "Final polish step processing complete.", agent_name=self.name, duration_ms=duration_step6_ms)


            # === Step 7: Clean and Assemble Final Output ===
            step7_name = "Assemble Final Document"
            self._log_step(step7_name, StepStatus.START.value, "Cleaning and assembling final document...", agent_name=self.name)
            final_doc_clean = self._clean_final_markdown(final_doc_raw)
            if not final_doc_clean:
                self._log_step(step7_name, StepStatus.WARN.value, "Final document empty after cleaning! Using pre-polish, cleaned content as fallback.", agent_name=self.name)
                final_doc_clean = self._clean_final_markdown(kb_doc_content)

            # Generate metadata comment
            contributing_agents_set = set()
            for agent in self.agents: # Iterate through all defined agents for the run
                if agent: contributing_agents_set.add(f"  - {agent.name} ({agent.role})")
            contributing_agents = "\n".join(sorted(list(contributing_agents_set)))

            # Construct final output with metadata
            final_markdown = (
                f"<!-- Orchestration ID: {self.orchestration_id} -->\n"
                f"<!-- Generated on: {release_date_str} -->\n"
                f"<!-- Contributing Agents:\n{contributing_agents}\n-->\n\n"
                f"{final_doc_clean}\n"
            )
            self._log_step(step7_name, StepStatus.END.value, "Final document assembly complete.", agent_name=self.name)

            # Save final markdown to file inside the orchestration output dir
            try:
                 async with aiofiles.open(final_md_file_path, "w", encoding="utf-8") as f:
                     await f.write(final_markdown)
                 self._log_step("Save Final Result", StepStatus.INFO.value, f"Saved final markdown to {final_md_file_path.relative_to(BASE_DIR)}", agent_name=self.name)
            except Exception as e:
                 # Log error but don't necessarily fail the whole orchestration here
                 self._log_step("Save Final Result", StepStatus.ERROR.value, f"Failed to save final markdown: {e}", agent_name=self.name)

            # Final Orchestration Success Log
            orch_duration_ms = int((time.monotonic() - self.start_time) * 1000)
            self._log_step(main_action_name, StepStatus.SUCCESS.value, f"Orchestration completed successfully.", agent_name=self.name, duration_ms=orch_duration_ms)
            orchestration_success = True # Mark as successful


        # --- Orchestration Error & Cancellation Handling ---
        except asyncio.CancelledError:
            orch_duration_ms = int((time.monotonic() - self.start_time) * 1000)
            self._log_step(main_action_name, StepStatus.CANCELLED.value, f"Orchestration was cancelled.", agent_name=self.name, duration_ms=orch_duration_ms)
            self._log_step(main_action_name, StepStatus.INFO.value, f"Attempting to cancel pending/in-progress tasks for cancelled orchestration...", agent_name=self.name)
            cancelled_count = 0
            # Use tracked task IDs for this specific run
            for task_id in self.tasks_created_this_run:
                try:
                     task = await self.task_manager._read_task_impl(task_id) # Read from cache
                     if task and task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                         await self.task_manager._update_task_impl(task_id, status=TaskStatus.CANCELLED, details=f"Orchestration {self.orchestration_id} cancelled.")
                         self._log_step(main_action_name, StepStatus.INFO.value, f"Marked task '{task_id}' as CANCELLED.", task_id=task_id, agent_name=self.name)
                         cancelled_count += 1
                except Exception as cancel_update_err:
                     logger.error(f"[{main_action_name}] Failed to update task '{task_id}' to CANCELLED status during cleanup: {cancel_update_err}")
            self._log_step(main_action_name, StepStatus.INFO.value, f"Cancellation cleanup finished. Marked {cancelled_count} tasks as CANCELLED.", agent_name=self.name)
            stats = await self._gather_statistics(time.monotonic())
            await self._save_output_metadata(final_result_file_name="<cancelled>", stats=stats)
            raise # Re-raise CancellationError

        except Exception as e:
            orch_duration_ms = int((time.monotonic() - self.start_time) * 1000)
            error_type = type(e).__name__
            self._log_step(main_action_name, StepStatus.ERROR.value, f"Orchestration failed: {error_type}: {e}", agent_name=self.name, duration_ms=orch_duration_ms)
            logger.exception(f"[{main_action_name}] Orchestration {self.orchestration_id} failed.")

            final_markdown = ( # Update error markdown
                f"# Orchestration Error\n\n"
                f"**Orchestration ID:** {self.orchestration_id}\n"
                f"**Error Type:** {error_type}\n"
                f"**Details:** {e}\n\n"
                f"Check logs (`steps.json`, `tasks.json`, console) and metadata file '{self.output_dir.relative_to(BASE_DIR)}/orchestration_output.json' for details."
            )
            self._log_step(main_action_name, StepStatus.INFO.value, f"Attempting to mark pending/in-progress tasks for failed orchestration as ERROR...", agent_name=self.name)
            error_marked_count = 0
            # Use tracked task IDs for this specific run
            for task_id in self.tasks_created_this_run:
                try:
                    task = await self.task_manager._read_task_impl(task_id) # Read from cache
                    if task and task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                        await self.task_manager._update_task_impl(task_id, status=TaskStatus.ERROR, details=f"Orchestration {self.orchestration_id} failed: {error_type}")
                        self._log_step(main_action_name, StepStatus.INFO.value, f"Marked task '{task_id}' as ERROR.", task_id=task_id, agent_name=self.name)
                        error_marked_count += 1
                except Exception as error_update_err:
                     logger.error(f"[{main_action_name}] Failed to update task '{task_id}' to ERROR status during error cleanup: {error_update_err}")
            self._log_step(main_action_name, StepStatus.INFO.value, f"Error cleanup finished. Marked {error_marked_count} tasks as ERROR.", agent_name=self.name)
            # Still try to save metadata even on failure
            stats = await self._gather_statistics(time.monotonic())
            await self._save_output_metadata(final_result_file_name="<error>", stats=stats)
            # Return the error markdown

        # --- Final Step: Save Metadata ---
        # This now happens after success or within error handling blocks
        if orchestration_success:
            try:
                stats = await self._gather_statistics(time.monotonic())
                await self._save_output_metadata(final_result_file_name=final_result_file_name, stats=stats)
            except Exception as meta_err:
                 self._log_step("Save Output Metadata", StepStatus.ERROR.value, f"Failed to save metadata after successful run: {meta_err}", agent_name=self.name)


        return final_markdown


# --- FastAPI Router Setup ---

# Create singleton instances of TaskManager and StepManager using shared log files
task_manager_instance = TaskManager(TASK_LOG_FILE)
step_manager_instance = StepManager(STEP_LOG_FILE)

async def get_task_manager() -> TaskManager:
    """FastAPI dependency for TaskManager."""
    if not task_manager_instance._initialized:
        await task_manager_instance.initialize() # Load initial cache
    return task_manager_instance

async def get_step_manager() -> StepManager:
    """FastAPI dependency for StepManager."""
    if not step_manager_instance._initialized:
        await step_manager_instance.initialize() # Ensure directory exists
    return step_manager_instance

# --- Gemini Client Dependency Injection ---
_gemini_model_client: Optional[GeminiModelClient] = None
_gemini_init_lock = asyncio.Lock()

async def get_gemini_client_dependency() -> Optional[GeminiModelClient]:
    """FastAPI dependency for the custom GeminiClient."""
    global _gemini_model_client
    if _gemini_model_client is not None: return _gemini_model_client
    async with _gemini_init_lock:
        if _gemini_model_client is not None: return _gemini_model_client
        if not GeminiClient:
            logger.warning("[Dependency] Gemini client requested, but `app.llm.gemini.GeminiClient` class not imported. Returning None.")
            return None
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("[Dependency] Gemini client requested, but GEMINI_API_KEY not set. Returning None.")
            return None
        logger.info("[Dependency] Attempting to initialize custom GeminiClient...")
        try:
            model_name = os.getenv("GEMINI_MODEL_NAME")
            client_args = {"api_key": api_key}
            if model_name:
                client_args["model"] = model_name
                logger.info(f"[Dependency] Initializing GeminiClient with model: '{model_name}'")
            else:
                 logger.info("[Dependency] Initializing GeminiClient with its default model.")
            client_instance = GeminiClient(**client_args)
            _gemini_model_client = client_instance
            logger.info("[Dependency] Custom GeminiClient dependency initialized successfully.")
            return _gemini_model_client
        except Exception as e:
            logger.error(f"[Dependency] Failed to initialize Custom GeminiClient: {e}", exc_info=True)
            _gemini_model_client = None
            return None

# --- API Endpoints ---

@router.get("/", response_class=PlainTextResponse, tags=["Orchestration"])
async def read_root(
    # --- Dependencies ---
    client: Optional[GeminiModelClient] = Depends(get_gemini_client_dependency),
    task_manager: TaskManager = Depends(get_task_manager),
    step_manager: StepManager = Depends(get_step_manager), # Inject StepManager
    # --- Query Parameters ---
    force_regenerate: bool = Query(False, description="Set to true to bypass the cache and force regeneration."),
    target_subfolder: Optional[str] = Query("app", description="Subfolder within the project root to analyze (e.g., 'app', 'src'). Defaults to 'app'. Set to '.' for project root."),
):
    """
    Main endpoint to trigger the automated documentation generation orchestration.

    Fetches project structure, uses AI agents to analyze and generate documentation,
    and returns the result as Markdown. Caching is used by default. Saves output and metadata
    to `.cache/output/<orchestration_id>/`.
    """
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    logger.info(f"[API /] Request {request_id} received. Force Regenerate: {force_regenerate}, Target Subfolder: '{target_subfolder}'")
    start_api_time = time.monotonic() # Track API request time

    # --- Critical Dependency Check: LLM Client ---
    if client is None:
        error_detail = "LLM service (custom GeminiClient) is unavailable or failed to initialize. Cannot proceed. Check server configuration (API key, imports) and logs."
        logger.error(f"[API /] Request {request_id}: Blocked. {error_detail}")
        # Log block step using the shared manager
        init_step = StepSchema(orchestration_id="N/A", agent_name="API", step_name="Orchestration Request", status=StepStatus.BLOCK, details=error_detail)
        asyncio.create_task(step_manager.create_step_log(init_step))
        return PlainTextResponse(content=f"# Service Unavailable\n\nRequest ID: {request_id}\n\n{error_detail}", status_code=503, media_type="text/plain")

    # --- Cache Handling ---
    use_cache = not force_regenerate and (os.getenv("USE_CACHE", "true").lower() == "true")
    if use_cache:
        if await is_cache_valid():
            cached_content = await load_from_cache()
            if cached_content:
                logger.info(f"[API /] Request {request_id}: Serving response from valid cache.")
                cache_hit_step = StepSchema(orchestration_id="N/A", agent_name="API", step_name="Cache Check", status=StepStatus.INFO, details="Cache hit, serving cached content.")
                asyncio.create_task(step_manager.create_step_log(cache_hit_step))
                return PlainTextResponse(content=cached_content, status_code=200, media_type="text/markdown")
            else:
                logger.warning(f"[API /] Request {request_id}: Cache file invalid/unreadable. Regenerating.")
        else:
             logger.info(f"[API /] Request {request_id}: Cache expired/not found. Regenerating.")
             cache_miss_step = StepSchema(orchestration_id="N/A", agent_name="API", step_name="Cache Check", status=StepStatus.INFO, details="Cache miss or invalid, regenerating.")
             asyncio.create_task(step_manager.create_step_log(cache_miss_step))
    else:
        logger.info(f"[API /] Request {request_id}: Cache explicitly bypassed or disabled. Regenerating.")
        cache_bypass_step = StepSchema(orchestration_id="N/A", agent_name="API", step_name="Cache Check", status=StepStatus.INFO, details="Cache bypassed via query or config.")
        asyncio.create_task(step_manager.create_step_log(cache_bypass_step))


    # --- Determine Target Folder ---
    try:
        if target_subfolder and target_subfolder != '.':
            target_folder_path = (BASE_DIR / target_subfolder).resolve()
            if BASE_DIR not in target_folder_path.parents and target_folder_path != BASE_DIR:
                 raise ValueError("Resolved target folder path is outside the project base directory.")
        elif target_subfolder == '.':
             target_folder_path = BASE_DIR.resolve() # Analyze project root
        else:
             target_folder_path = (BASE_DIR / "app").resolve() # Default to 'app' subfolder

        logger.info(f"[API /] Request {request_id}: Target folder for analysis: '{target_folder_path}'")
        if not target_folder_path.is_dir():
             raise ValueError(f"Target path '{target_folder_path}' is not a valid directory.")

    except (ValueError, Exception) as path_err:
         error_detail = f"Invalid target subfolder specified: '{target_subfolder}'. Error: {path_err}"
         logger.error(f"[API /] Request {request_id}: Configuration error. {error_detail}")
         folder_err_step = StepSchema(orchestration_id="N/A", agent_name="API", step_name="Target Folder Validation", status=StepStatus.ERROR, details=error_detail)
         asyncio.create_task(step_manager.create_step_log(folder_err_step))
         raise HTTPException(status_code=400, detail=error_detail)

    # --- Orchestration Execution ---
    manager = None # Initialize to None
    orchestration_id = "N/A" # Default if manager creation fails
    try:
        # Each request gets a NEW manager instance for isolation
        manager = Manager(client=client, task_manager=task_manager, step_manager=step_manager) # Pass managers
        orchestration_id = manager.orchestration_id # Get ID for logging
        logger.info(f"[API /] Request {request_id}: Starting orchestration run {orchestration_id}...")
        start_orch_time = time.monotonic()
        final_markdown = await manager.orchestrate(str(target_folder_path))
        orch_duration = time.monotonic() - start_orch_time

        # --- Process Result ---
        is_known_error_output = final_markdown and (
            "# Orchestration Error" in final_markdown or
            "# Agent Role Error" in final_markdown or
            "# Agent Assignment Error" in final_markdown or
            "# Agent Initialization Error" in final_markdown or
            "# Configuration Error" in final_markdown
            )

        if not is_known_error_output:
            logger.info(f"[API /] Request {request_id}: Orchestration {orchestration_id} successful ({orch_duration:.2f}s).")
            status_code = 200
            media_type = "text/markdown"
            if use_cache: await save_to_cache(final_markdown)
        else:
            logger.error(f"[API /] Request {request_id}: Orchestration {orchestration_id} finished with an error ({orch_duration:.2f}s). Returning error details.")
            status_code = 500 # Internal server error, as the orchestration failed
            media_type = "text/markdown" # Return error details as markdown

        total_api_duration = time.monotonic() - start_api_time
        logger.info(f"[API /] Request {request_id}: Sending response. Status: {status_code}, Media-Type: {media_type}, Length: {len(final_markdown)}, Total API Time: {total_api_duration:.2f}s")
        return PlainTextResponse(content=final_markdown, status_code=status_code, media_type=media_type)

    except asyncio.CancelledError:
        total_api_duration = time.monotonic() - start_api_time
        logger.warning(f"[API /] Request {request_id}: Orchestration task {orchestration_id} cancelled ({total_api_duration:.2f}s API time).")
        # Manager's error handling should have logged steps and saved metadata.
        raise # Re-raise

    except HTTPException as http_exc:
        total_api_duration = time.monotonic() - start_api_time
        logger.warning(f"[API /] Request {request_id}: Caught HTTPException: {http_exc.status_code} - {http_exc.detail} ({total_api_duration:.2f}s API time)")
        http_err_step = StepSchema(orchestration_id=orchestration_id, agent_name="API", step_name="HTTP Exception", status=StepStatus.ERROR, details=f"{http_exc.status_code}: {http_exc.detail}")
        asyncio.create_task(step_manager.create_step_log(http_err_step))
        raise http_exc

    except Exception as e:
        total_api_duration = time.monotonic() - start_api_time
        error_type = type(e).__name__
        logger.exception(f"[API /] Request {request_id}: Unhandled exception during orchestration {orchestration_id} ({total_api_duration:.2f}s API time): {e}")
        if not manager: # Log error if manager creation failed
            unhandled_err_step = StepSchema(orchestration_id=orchestration_id, agent_name="API", step_name="Unhandled Exception", status=StepStatus.ERROR, details=f"Unhandled {error_type} during manager init or early orchestration: {e}")
            asyncio.create_task(step_manager.create_step_log(unhandled_err_step))

        error_content = f"# Internal Server Error\n\nRequest ID: {request_id}\nOrchestration ID: {orchestration_id}\nError: {error_type}. Check server logs and shared log files (.cache/)."
        return PlainTextResponse(content=error_content, status_code=500, media_type="text/plain")


# --- Task API Endpoints ---
@router.get("/tasks", response_model=List[TaskSchema], summary="List All Tasks", tags=["Tasks"])
async def get_tasks_api(
    status: Optional[TaskStatus] = Query(None, description=f"Filter by status (e.g., {', '.join(s.value for s in TaskStatus)})"),
    assignee: Optional[str] = Query(None, description="Filter by assigned agent name"),
    orchestration_id: Optional[str] = Query(None, description="Filter by orchestration ID"),
    limit: int = Query(100, description="Max tasks to return", gt=0, le=1000),
    sort_by: str = Query("created_at", description="Sort field (created_at, updated_at, name, status)", regex="^(created_at|updated_at|name|status)$"),
    sort_order: str = Query("desc", description="Sort order ('asc' or 'desc')", regex="^(asc|desc)$"),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Retrieves a list of tasks from the manager's cache, optionally filtered and sorted."""
    filter_log = f"Filters: status={status.value if status else 'Any'}, assignee={assignee or 'Any'}, orchID={orchestration_id or 'Any'}"
    sort_log = f"Sort: {sort_by} {sort_order}, Limit: {limit}"
    logger.info(f"[API /tasks] Request received. {filter_log}. {sort_log}")
    try:
        # Read from the manager's current in-memory cache
        tasks = await task_manager._read_all_tasks_impl(
            status_filter=status,
            assigned_filter=assignee,
            orchestration_id_filter=orchestration_id
        )
        reverse_sort = sort_order.lower() == "desc"
        try:
            tasks.sort(key=lambda t: getattr(t, sort_by), reverse=reverse_sort)
        except AttributeError:
             logger.warning(f"[API /tasks] Invalid sort_by field '{sort_by}'. Using default sort (created_at desc).")
             tasks.sort(key=lambda t: t.created_at, reverse=True) # Default sort

        limited_tasks = tasks[:limit]
        logger.info(f"[API /tasks] Returning {len(limited_tasks)} tasks from cache.")
        return limited_tasks
    except ValueError as ve:
         logger.error(f"[API /tasks] Invalid filter value: {ve}")
         raise HTTPException(status_code=400, detail=f"Invalid filter value provided: {ve}")
    except Exception as e:
        logger.error(f"[API /tasks] Error reading tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read tasks.")

@router.get("/tasks/{task_id}", response_model=TaskSchema, summary="Get Task by ID", tags=["Tasks"])
async def get_task_api(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Retrieves details of a specific task by ID from the manager's cache."""
    logger.info(f"[API /tasks/{task_id}] Request received.")
    task = await task_manager._read_task_impl(task_id)
    if task:
        logger.debug(f"[API /tasks/{task_id}] Task found in cache.")
        return task
    else:
        logger.warning(f"[API /tasks/{task_id}] Task not found in cache.")
        # Optionally: Could try to reload the main task file here?
        # await task_manager.initialize()
        # task = await task_manager._read_task_impl(task_id)
        # if task: return task
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found in local cache.")


# --- Step Log API Endpoint ---
@router.get("/steps", response_model=List[StepSchema], summary="List Orchestration Step Logs", tags=["Steps"])
async def get_steps_api(
    orchestration_id: Optional[str] = Query(None, description="Filter steps by a specific orchestration ID"),
    limit: int = Query(1000, description="Maximum number of step logs to return", gt=0, le=5000),
    step_manager: StepManager = Depends(get_step_manager) # Inject StepManager
):
    """Retrieves a list of orchestration step logs from the shared log file, optionally filtered."""
    filter_log = f"Filters: orchID={orchestration_id or 'All'}, Limit={limit}"
    logger.info(f"[API /steps] Request received. {filter_log}")
    try:
        # Use the StepManager's get method which reads/filters from file
        steps = await step_manager.get_steps(orchestration_id=orchestration_id, limit=limit)
        logger.info(f"[API /steps] Returning {len(steps)} step logs.")
        return steps
    except Exception as e:
        logger.error(f"[API /steps] Error reading step logs from shared file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read step logs.")


# --- Agent Memory API Endpoint ---
class AgentMemoryResponse(BaseModel):
    """Response schema for agent memory details."""
    agent_name: str
    agent_role: str
    goal: Optional[str] = None
    backstory: Optional[str] = None
    memory: MemorySchema # Embed the full MemorySchema (reflects IN-RAM state)

async def get_observation_manager(
    client: Optional[GeminiModelClient] = Depends(get_gemini_client_dependency),
    task_manager: TaskManager = Depends(get_task_manager),
    step_manager: StepManager = Depends(get_step_manager)
) -> Optional[Manager]:
    """Dependency to create a temporary Manager instance for observation."""
    if client:
        try:
            # Create temporary manager - its state is isolated.
            temp_manager = Manager(client=client, task_manager=task_manager, step_manager=step_manager)
            logger.debug("[Dependency/ObservationManager] Created temporary Manager for observation.")
            return temp_manager
        except Exception as e:
            logger.error(f"[Dependency/ObservationManager] Failed to create temporary Manager: {e}", exc_info=True)
            return None
    else:
        logger.warning("[Dependency/ObservationManager] LLM client unavailable, cannot create temporary Manager.")
        return None

@router.get("/agents/memory", response_model=List[AgentMemoryResponse], summary="View Defined Agent States", tags=["Agents"])
async def get_agent_memories(
    manager: Optional[Manager] = Depends(get_observation_manager) # Use observation manager
):
    """
    Retrieves the state (including name, role, goal, backstory, and in-memory state)
    from agents defined within a TEMPORARY Manager instance. Primarily shows the
    initial state (empty history/scratchpad). Does NOT reflect memory of actively
    running agents from the '/' endpoint.
    """
    logger.info("[API /agents/memory] Request received (observational).")
    if not manager:
         logger.warning("[API /agents/memory] Observation Manager unavailable. Returning empty list.")
         return []
    if not manager.agents:
         logger.warning("[API /agents/memory] Observation Manager instance has no agents defined. Returning empty list.")
         return []

    agent_states = []
    try:
        for agent in manager.agents:
            mem_copy = agent.memory.model_copy(deep=True) # Deep copy of memory state
            agent_states.append( AgentMemoryResponse(
                agent_name=agent.name,
                agent_role=agent.role,
                goal=agent.goal, # Include goal
                backstory=agent.backstory, # Include backstory
                memory=mem_copy # Include the memory snapshot
            ))
        logger.info(f"[API /agents/memory] Returning state for {len(agent_states)} defined agents (from Observation Manager ID: {manager.orchestration_id}).")
        return agent_states
    except Exception as e:
        logger.error(f"[API /agents/memory] Error retrieving agent states: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent states: {e}")

# --- Orchestration Output Metadata API Endpoint ---
@router.get("/orchestrations/{orchestration_id}/output", response_model=OrchestrationOutputSchema, summary="Get Orchestration Output Metadata", tags=["Orchestration"])
async def get_orchestration_output(
    orchestration_id: str,
):
    """
    Retrieves the saved output metadata JSON for a specific orchestration run
    from its dedicated output directory: `.cache/output/<orchestration_id>/orchestration_output.json`.
    """
    logger.info(f"[API /orchestrations/{orchestration_id}/output] Request received.")
    # Construct the path to the specific metadata file
    output_dir = OUTPUT_BASE_DIR / orchestration_id
    metadata_file = output_dir / "orchestration_output.json"

    if not metadata_file.is_file(): # Check if it's a file
        logger.warning(f"[API /orchestrations/{orchestration_id}/output] Metadata file not found or is not a file: {metadata_file}")
        raise HTTPException(status_code=404, detail=f"Output metadata for orchestration ID '{orchestration_id}' not found. Ensure the orchestration has run and saved its metadata.")

    try:
        async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
            content = await f.read()
            metadata_dict = json.loads(content)
        # Validate the loaded data against the Pydantic schema
        validated_metadata = OrchestrationOutputSchema(**metadata_dict)
        logger.info(f"[API /orchestrations/{orchestration_id}/output] Returning metadata.")
        return validated_metadata
    except json.JSONDecodeError as json_err:
        logger.error(f"[API /orchestrations/{orchestration_id}/output] Error decoding metadata JSON from {metadata_file}: {json_err}")
        raise HTTPException(status_code=500, detail=f"Failed to decode metadata for orchestration '{orchestration_id}'. File might be corrupt.")
    except (IOError, ValueError, TypeError) as e: # Catch Pydantic validation errors too
        logger.error(f"[API /orchestrations/{orchestration_id}/output] Error reading or validating metadata file {metadata_file}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read or validate metadata for orchestration '{orchestration_id}': {e}")
    except Exception as e:
         logger.error(f"[API /orchestrations/{orchestration_id}/output] Unexpected error processing metadata file {metadata_file}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Unexpected error processing metadata for orchestration '{orchestration_id}'.")