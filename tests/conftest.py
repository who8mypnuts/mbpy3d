import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYMB3D_CACHE", str(ROOT / ".pytest_cache" / "pymb3d"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def cpu_backend():
    from pymb3d.backends.cpu_backend import CpuBackend

    return CpuBackend()


@pytest.fixture(scope="session")
def opencl_backend():
    from pymb3d.backends.opencl_backend import OpenCLBackend, list_opencl_devices

    if not list_opencl_devices():
        pytest.skip("no OpenCL device")
    return OpenCLBackend()


@pytest.fixture(scope="session")
def opengl_backend():
    from pymb3d.backends.base import BackendError
    from pymb3d.backends.opengl_backend import OpenGLBackend

    be = OpenGLBackend()
    try:
        be.ensure_context()
    except BackendError as e:
        pytest.skip(f"no OpenGL context: {e}")
    return be
