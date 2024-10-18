from modelforge.custom_types import ModelType
from modelforge.dataset.dataset import NNPInput


def visualize_model(nnp_input: NNPInput, potential_name: ModelType):
    # visualize the compute graph
    from modelforge.utils.io import import_
    from modelforge.tests.helper_functions import setup_potential_for_test

    torchviz = import_("torchviz")

    inference_model = setup_potential_for_test(
        potential_name,
        "inference",
    )

    yhat = inference_model(nnp_input)["per_system_energy"]
    torchviz.make_dot(
        yhat,
        params=dict(list(inference_model.named_parameters())),
        show_attrs=True,
        show_saved=True,
    ).render(f"compute_graph_{potential_name}1", format="png")
