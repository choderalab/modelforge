from modelforge.utils.prop import NNPInput


def nnpinput_flatten(nnpinput: NNPInput):
    # Collect all attributes into a tuple
    children = (
        nnpinput.atomic_numbers,
        nnpinput.positions,
        nnpinput.atomic_subsystem_indices,
        nnpinput.per_system_total_charge,
        nnpinput.box_vectors,
        nnpinput.per_system_spin_state,
        nnpinput.is_periodic,
        nnpinput.pair_list,
        nnpinput.per_atom_partial_charge,
    )
    # No auxiliary data is needed
    aux_data = None
    return (children, aux_data)


def nnpinput_unflatten(aux_data, children):
    # Reconstruct the NNPInput instance from the children
    return NNPInput(*children)


def convert_NNPInput_to_jax(nnp_input: NNPInput):
    """
    Convert the NNPInput to a JAX-compatible format.
    """
    from modelforge.utils.io import import_

    convert_to_jax = import_("pytorch2jax").pytorch2jax.convert_to_jax

    nnp_input.atomic_numbers = convert_to_jax(nnp_input.atomic_numbers)
    nnp_input.positions = convert_to_jax(nnp_input.positions)
    nnp_input.atomic_subsystem_indices = convert_to_jax(
        nnp_input.atomic_subsystem_indices
    )
    nnp_input.per_system_total_charge = convert_to_jax(
        nnp_input.per_system_total_charge
    )
    nnp_input.box_vectors = convert_to_jax(nnp_input.box_vectors)
    nnp_input.is_periodic = convert_to_jax(nnp_input.is_periodic)

    nnp_input.pair_list = convert_to_jax(nnp_input.pair_list)
    nnp_input.per_atom_partial_charge = convert_to_jax(
        nnp_input.per_atom_partial_charge
    )
    nnp_input.per_system_spin_state = convert_to_jax(nnp_input.per_system_spin_state)

    return nnp_input
