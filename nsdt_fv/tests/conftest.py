import pytest

from nsdt_fv.data.stream import build_stream


@pytest.fixture(scope="session")
def stream():
    return build_stream(0)
