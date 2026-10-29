"""Visual themes: palettes applied to the whole application (MB3D-rs has
theme files; here a few built-in looks plus the system look) and the
interface scale."""
from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

THEMES = {
    "system": None,
    "glossy (dark)": {"window": "#2b2b2b", "base": "#1e1e1e", "text": "#e6e6e6", "button": "#3a3a3a",
                      "highlight": "#4a90d9", "link": "#6fb3ff", "mid": "#555555"},
    "graphite": {"window": "#3c3f41", "base": "#2b2b2b", "text": "#dcdcdc", "button": "#4a4d4f",
                 "highlight": "#5a8fc8", "link": "#8ab4e8", "mid": "#606366"},
    "midnight": {"window": "#1b2233", "base": "#121826", "text": "#d8e0f0", "button": "#26304a",
                 "highlight": "#3d7ae0", "link": "#7fb0ff", "mid": "#3a4766"},
    "ember": {"window": "#2a2522", "base": "#1c1815", "text": "#ecdcd0", "button": "#3b332e",
              "highlight": "#e0742c", "link": "#ffa65c", "mid": "#5a4b42"},
    "forest": {"window": "#243028", "base": "#17201a", "text": "#d9e6da", "button": "#2f4034",
               "highlight": "#4caf6a", "link": "#8fe0a6", "mid": "#3f5546"},
    "daylight": {"window": "#f0f0ee", "base": "#ffffff", "text": "#202020", "button": "#e4e4e0",
                 "highlight": "#2e8b9a", "link": "#1f6f7a", "mid": "#b0b0b0"},
    "high contrast": {"window": "#000000", "base": "#000000", "text": "#ffffff", "button": "#202020",
                      "highlight": "#ffd400", "link": "#ffd400", "mid": "#808080"},
}


def theme_names() -> list[str]:
    return list(THEMES)


def apply_theme(app: QApplication, name: str) -> None:
    spec = THEMES.get(name)
    if spec is None:
        app.setStyle("Fusion")
        app.setPalette(app.style().standardPalette())
        return
    app.setStyle("Fusion")
    p = QPalette()
    window, base, text = QColor(spec["window"]), QColor(spec["base"]), QColor(spec["text"])
    button, highlight, mid = QColor(spec["button"]), QColor(spec["highlight"]), QColor(spec["mid"])
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, window)
    p.setColor(QPalette.ColorRole.ToolTipBase, base)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, button)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ff5555"))
    p.setColor(QPalette.ColorRole.Link, QColor(spec["link"]))
    p.setColor(QPalette.ColorRole.Highlight, highlight)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff") if name != "high contrast" else QColor("#000000"))
    p.setColor(QPalette.ColorRole.Mid, mid)
    p.setColor(QPalette.ColorRole.Dark, mid.darker(150))
    p.setColor(QPalette.ColorRole.Light, mid.lighter(150))
    disabled = text.darker(200) if text.lightness() > 128 else text.lighter(200)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled)
    app.setPalette(p)


def saved_theme(settings: QSettings) -> str:
    return str(settings.value("theme", "glossy (dark)"))


def apply_saved(app: QApplication, settings: QSettings) -> str:
    name = saved_theme(settings)
    if name not in THEMES:
        name = "glossy (dark)"
    apply_theme(app, name)
    return name
