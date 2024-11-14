from fastworkflow.command_executor import CommandResponse, Action
from fastworkflow.session import Session

from ..parameter_extraction.signatures import CommandParameters
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: CommandParameters
    ) -> list[CommandResponse]:
        output = process_command(session, command_parameters)

        return [
            CommandResponse(
                response=(
                    f"move to workitem succeeded: {output.target_workitem_found}\n"
                    f"active workitem status: {output.status_of_target_workitem}"
                ),
                next_actions=[
                    Action(
                        session_id=session.id,
                        workitem_type="sample_workflow",
                        command_name="move_to_workitem",
                        command="Move to mytask",
                        parameters={"workitem_path": "mytask", "workitem_id": None},
                    )
                ],
            )
        ]


# if __name__ == "__main__":
