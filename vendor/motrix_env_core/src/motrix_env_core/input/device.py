# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Input-device type hierarchy and device-specific query contracts."""

from abc import ABC, abstractmethod


class InputDevice(ABC):
    """Nominal base for input devices."""


class KeyboardDevice(InputDevice, ABC):
    """Event-framed keyboard input."""

    @abstractmethod
    def poll(self) -> None:
        """Freeze the keyboard event frame used by subsequent queries."""

    @abstractmethod
    def is_key_down(self, key: str) -> bool:
        """Return whether a press edge occurred in the current event frame."""

    @abstractmethod
    def is_key_up(self, key: str) -> bool:
        """Return whether a release edge occurred in the current event frame."""

    @abstractmethod
    def is_pressing(self, key: str) -> bool:
        """Return whether the key is held at the end of the current event frame."""


class GamePadDevice(InputDevice, ABC):
    """Event-framed gamepad input."""

    @abstractmethod
    def poll(self) -> None:
        """Freeze the gamepad event frame used by subsequent queries."""

    @abstractmethod
    def axis_value(self, axis: str) -> float:
        """Return one finite, normalized axis value inside ``[-1, 1]``."""

    @abstractmethod
    def is_button_down(self, button: str) -> bool:
        """Return whether a button press edge occurred in the current event frame."""

    @abstractmethod
    def is_button_up(self, button: str) -> bool:
        """Return whether a button release edge occurred in the current event frame."""

    @abstractmethod
    def is_button_pressing(self, button: str) -> bool:
        """Return whether a button is held at the end of the current event frame."""


__all__ = ["GamePadDevice", "InputDevice", "KeyboardDevice"]
