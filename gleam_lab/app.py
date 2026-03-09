from __future__ import annotations

import os
from packaging.version import Version

from isaaclab.app import AppLauncher


URDF_IMPORTER_ISAACSIM_51 = "isaacsim.asset.importer.urdf-2.4.31"
URDF_IMPORTER_DEFAULT = "isaacsim.asset.importer.urdf"


def launch_app(headless: bool = True, enable_cameras: bool = True):
    """Launch Isaac Sim once for the current process."""
    launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
    return launcher.app


def enable_urdf_importer_for_isaacsim_51() -> str:
    """Enable the pinned URDF importer used by Isaac Lab tests on Isaac Sim 5.1+."""
    import omni.kit.app
    from isaaclab.utils.version import get_isaac_sim_version

    manager = omni.kit.app.get_app().get_extension_manager()
    if get_isaac_sim_version() >= Version("5.1"):
        extension_name = URDF_IMPORTER_ISAACSIM_51
    else:
        extension_name = URDF_IMPORTER_DEFAULT
    manager.set_extension_enabled_immediate(extension_name, True)
    return extension_name


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
