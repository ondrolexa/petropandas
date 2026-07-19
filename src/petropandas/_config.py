"""Global configuration for petropandas."""

from __future__ import annotations


class PPConfig:
    """Mutable configuration object for petropandas defaults.

    Access via the module-level ``ppconfig`` singleton::

        from petropandas import ppconfig

        ppconfig.default_db = "ig"
        ppconfig.reset()  # back to defaults
    """

    default_system: str = "MnNCKFMASHTO"
    default_oxygen: float = 0.01
    default_H2O: float = -1.0
    default_db: str = "mp"
    default_sys_in: str = "mol"

    def reset(self) -> None:
        """Reset all settings to their original defaults."""
        self.default_system = "MnNCKFMASHTO"
        self.default_oxygen = 0.01
        self.default_H2O = -1.0
        self.default_db = "mp"
        self.default_sys_in = "mol"


ppconfig = PPConfig()
