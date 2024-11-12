import os

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

from fastworkflow.session import Session


class SemanticRouterDefinition:
    def __init__(self, encoder: HuggingFaceEncoder, workflow_folderpath: str):
        self._encoder = encoder
        self._route_layers_folderpath = os.path.join(
            workflow_folderpath, "___route_layers"
        )

    def get_route_layer(self, workitem_type: str) -> RouteLayer:
        route_layer_filepath = os.path.join(
            self._route_layers_folderpath, f"{workitem_type}.json"
        )
        return RouteLayer.from_json(route_layer_filepath)

    def train(self, session: Session):
        for workitem_type in session.workflow_definition.types:
            command_names = session.command_routing_definition.get_command_names(
                workitem_type
            )

            utterance_command_tuples = []

            routes = []
            for command_name in command_names:
                utterances = session.utterance_definition.get_command_utterances(
                    workitem_type, command_name
                )
                utterances_func = utterances.get_generated_utterances_func(
                    session.workflow_folderpath
                )
                utterance_list = utterances_func(session)

                utterance_command_tuples.extend(
                    list(zip(utterance_list, [command_name] * len(utterance_list)))
                )

                routes.append(Route(name=command_name, utterances=utterance_list))

            rl = RouteLayer(encoder=self._encoder, routes=routes)

            # unpack the test data
            X, y = zip(*utterance_command_tuples)
            # evaluate using the default thresholds
            accuracy = rl.evaluate(X=X, y=y)
            print(f"{workitem_type}: Accuracy before training: {accuracy*100:.2f}%")

            threshold_accuracy = 0.1    #TODO: why is training not working?
            if accuracy <= threshold_accuracy:
                # Call the fit method
                rl.fit(X=X, y=y)
                # route_thresholds = rl.get_thresholds()
                accuracy = rl.evaluate(X=X, y=y)
                print(f"{workitem_type}: Accuracy after training: {accuracy*100:.2f}%")
            else:
                print(
                    f"{workitem_type}: Accuracy exceeds {threshold_accuracy*100:.2f}%. No training necessary."
                )

            # save to JSON
            rl.to_json(
                os.path.join(self._route_layers_folderpath, f"{workitem_type}.json")
            )
