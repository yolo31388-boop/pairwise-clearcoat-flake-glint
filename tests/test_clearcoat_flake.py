import pytest
import graphics.clearcoat_flake as mod

CASES = [
    ("grazing_fresnel", (1,)),
    ("coat_normal_independent", (1,)),
    ("tangent_flake", (1,)),
    ("layered_gbuffer", (1,)),
    ("probe_refresh", (1,)),
    ("queue_split", (1,)),
    ("load_paint", (1,)),
    ("shade_paint", (1,)),
    ("resolve_paint", (1,)),
    ("present_paint", (1,)),
    ("validate_paint", (1,)),
    ("metrics_paint", (1,)),
    ("audit_paint", (1,)),
    ("fallback_paint", (1,)),
    ("determinism_paint", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
