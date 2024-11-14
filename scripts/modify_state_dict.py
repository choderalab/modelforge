"""
Script to modify a state dict to include only_unique_pairs dictionary key.

This is only necessary for models trained prior to PR #299 in modelforge, that provides
integration with OpenMM and some refactoring of the neighborlisting schemes.


"""


def modify_state_dict(
    state_dict_input_file_path: str,
    state_dict_output_file_path: str,
    only_unique_pairs: bool,
):
    """
    Modify a state dict to include the only_unique_pairs dictionary key.

    Parameters
    ----------
    state_dict_input_file_path: str
        Input file with path to the input state dict file
    state_dict_output_file_path: str
        Output file with path to the output state dict file
    only_unique_pairs: bool
        Boolean value to set the only_unique_pairs key for the neighborlist
        This value should be True for the ANI models, False for most other models.

    Returns
    -------

    """
    import torch

    # Load the state dict
    state_dict = torch.load(state_dict_input_file_path)

    # Set the only_unique_pairs key
    state_dict["neighborlist.only_unique_pairs"] = torch.Tensor([only_unique_pairs])

    # Save the modified state dict
    torch.save(state_dict, state_dict_output_file_path)
