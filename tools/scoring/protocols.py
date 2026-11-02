"""Final native scoring contracts, without alternative answer extraction rules."""
from copy import deepcopy

PROTOCOLS = {
    'minigrid': dict(
        family='ssot', role='final_native_reproduction', script='tools/scoring/minigrid.py',
        parser='Original FINAL SSOT scorer and exam_schema, including original formatting recovery',
        iterations=None, seed=None,
        b_weights=dict(agent_pos=.30, agent_dir=0., carrying=.15, front_cell=.25, objects_jaccard=.20),
        b_weight_denominator=.90, b_accuracy_threshold=.90),
    'structured_cot': dict(
        family='structured', role='final_native_reproduction',
        script='tools/scoring/structured_cot.py',
        parser='Final structured-CoT native parser; strict and recoverable answers both contribute',
        iterations=20000, seed=20260724,
        seed_derivation='Native stable_seed(base_seed, model, prompt_arm, task); native paired-delta seeds'),
    'oracle_state': dict(
        family='oracle_state', role='final_native_reproduction',
        script='tools/scoring/oracle_state.py',
        parser='Final native four-field parser, including original recoverable field rules',
        iterations=20000, seed=20260725,
        seed_derivation='Native condition/task/model seed derivation'),
    'habitat': dict(
        family='habitat', role='final_native_reproduction',
        script='tools/scoring/habitat.py',
        parser='Original native Habitat last-option/last-verdict-keyword parser',
        iterations=None, seed=None),
    'miniworld_a': dict(
        family='miniworld', task='A', role='final_native_reproduction',
        script='tools/scoring/miniworld_a.py',
        parser='Native MiniWorld A strict/recoverable option parser', iterations=None, seed=None),
    'miniworld_c': dict(
        family='miniworld', task='C', role='final_native_reproduction',
        script='tools/scoring/miniworld_c.py',
        parser='Native MiniWorld C exam_schema verdict parser', iterations=None, seed=None),
    'miniworld_d': dict(
        family='miniworld', task='D', role='final_native_reproduction',
        script='tools/scoring/miniworld_d.py',
        parser='Native MiniWorld trajectory D verdict parser', iterations=None, seed=None),
    'miniworld_reference': dict(
        family='miniworld', task='D-reference', role='final_native_reproduction',
        script='tools/scoring/miniworld_reference.py',
        parser='Native MiniWorld reference-family D verdict parser', iterations=None, seed=None),
}

PROTOCOL_ALIASES = {
    'historical_ssot_v1': 'minigrid',
    'structured_native_v1': 'structured_cot',
    'oracle_state_v4_native_v1': 'oracle_state',
    'habitat_native_v1': 'habitat',
    'miniworld_a_native_v1': 'miniworld_a',
    'miniworld_c_native_v1': 'miniworld_c',
    'miniworld_d_native_v1': 'miniworld_d',
    'miniworld_reference_native_v1': 'miniworld_reference',
}


def canonical_protocol(protocol: str) -> str:
    """Resolve accepted aliases to the public protocol name."""
    return PROTOCOL_ALIASES.get(protocol, protocol)


def describe(protocol):
    """Return the native contract; invocation overrides are recorded separately."""
    protocol = canonical_protocol(protocol)
    if protocol not in PROTOCOLS:
        raise ValueError('Unknown native scoring protocol: ' + str(protocol))
    return dict(protocol_id=protocol, contract_version=1, **deepcopy(PROTOCOLS[protocol]))
