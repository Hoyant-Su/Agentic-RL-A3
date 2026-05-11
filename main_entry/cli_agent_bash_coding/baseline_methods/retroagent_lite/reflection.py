import json
import re
from dataclasses import dataclass

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

@dataclass(frozen=True)
class ParsedReflection:
    raw_json: str
    subtasks: list[dict]
    task_success: bool | None
    action_lesson: str
    navigation_lesson: str

    @property
    def total_subtasks(self) -> int:
        return len(self.subtasks)

    @property
    def completed_subtasks(self) -> int:
        total = 0
        for item in self.subtasks:
            status = str(item.get("status", "")).strip().lower()
            if status == "completed":
                total += 1
        return total

    @property
    def completed_ratio(self) -> float:
        if self.total_subtasks == 0:
            return 0.0
        return float(self.completed_subtasks) / float(self.total_subtasks)

    @property
    def lesson_text(self) -> str:
        lessons = []
        if self.action_lesson:
            lessons.append(f"Action Insight: {self.action_lesson}")
        if self.navigation_lesson:
            lessons.append(f"Navigation Insight: {self.navigation_lesson}")
        return " | ".join(lessons)

def build_reflection_prompt(
    task_description: str, trajectory_text: str, success: bool
) -> str:
    _ = success
    return (
        "You are an expert evaluating a shell coding agent attempt.\n"
        f"Target Task: {task_description}\n\n"
        "You have just completed an attempt at this shell coding task.\n"
        "Trajectory of the attempt:\n"
        f"{trajectory_text}\n\n"
        "<think>\n"
        "If a reference trajectory exists, compare it with the current trajectory.\n"
        "Analyze the trajectory to determine if the task was successful:\n"
        "1. Identify the specific requirements in the 'Target Task' (relevant files, required edits, checks, or final answer).\n"
        "2. Examine the sequence of actions. Did the agent successfully locate the relevant files or information?\n"
        "3. If file editing or command execution was required, were the correct commands or edits used?\n"
        "4. Did the agent produce the correct final file state or final answer?\n"
        "5. Did the trajectory end with the 'submit_answer' action after achieving the goal state? (If the agent stopped prematurely or failed to submit the final answer, it is a failure).\n"
        "6. What specific actions or decisions led to this outcome?\n"
        "7. What are the 1-2 most valuable lessons from this attempt?\n"
        "</think>\n\n"
        "Output your evaluation as JSON:\n\n"
        "{{\n"
        '"subtasks": [\n'
        '{{"name": "locate_artifact", "description": "[describe finding the relevant file, command, or evidence]", "status": "[completed or incomplete]"}},\n'
        '{{"name": "inspect_state", "description": "[describe inspecting the current file system or outputs]", "status": "[completed or incomplete]"}},\n'
        '{{"name": "modify_state", "description": "[describe editing files or executing commands if applicable, else \'N/A\']", "status": "[completed, incomplete, or N/A]"}},\n'
        '{{"name": "submit_result", "description": "[describe the final answer or final state submission]", "status": "[completed or incomplete]"}}\n'
        "],\n"
        "\"task_success\": [true if the correct final state or final answer was achieved and 'submit_answer' was called, false otherwise],\n"
        "\"action_lesson\": \"[key action insight, e.g., 'Used grep to locate the target string before editing' OR 'Edited the wrong file before verifying the path']\"\n"
        ",\n"
        "\"navigation_lesson\": \"[workspace/search insight, e.g., 'Systematically inspected the relevant directory before editing' OR 'Wasted steps revisiting unrelated files']\"\n"
        "}}\n\n"
        "EVALUATION GUIDELINES:\n"
        "- **Determine Success Yourself:** You must judge 'task_success' by comparing the final state in the trajectory to the Target Task.\n"
        "- **Criteria for Success:** The task is ONLY true if the agent identified the correct target, made the correct edits or checks, achieved the correct final state or answer, and issued the 'submit_answer' action.\n"
        "- **Criteria for Failure:** If the trajectory ends without the 'submit_answer' command, or if the agent submitted an answer without completing the goal, 'task_success' is false.\n"
        "- Each subtask status must reflect actual trajectory events.\n"
        "- Lessons should explain factors that led to the outcome.\n"
        "- Reference specific elements from trajectory (file paths, commands, outputs, or observations).\n"
        "- Use null for lessons only if truly not applicable.\n\n"
        "Output ONLY the JSON evaluation.\n"
    )

def parse_reflection(text: str) -> ParsedReflection | None:
    match = _JSON_BLOCK.search(text)
    if match is None:
        return None
    raw_json = match.group(0)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list):
        subtasks = []
    task_success = data.get("task_success", None)
    if not isinstance(task_success, bool):
        task_success = None
    action_lesson = data.get("action_lesson", None)
    navigation_lesson = data.get("navigation_lesson", None)
    return ParsedReflection(
        raw_json=raw_json,
        subtasks=[item for item in subtasks if isinstance(item, dict)],
        task_success=task_success,
        action_lesson="" if action_lesson is None else str(action_lesson).strip(),
        navigation_lesson=""
        if navigation_lesson is None
        else str(navigation_lesson).strip(),
    )
