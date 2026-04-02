import QtQuick
import QtQuick.Controls
import "."

// Reusable styled text field matching the app's design system.
// Supports all standard TextField properties (placeholderText, echoMode,
// maximumLength, onAccepted, etc.) — set them at the call site.
TextField {
    id: control

    font.pixelSize: 13
    color: AppPalette.textPrimary
    selectionColor: AppPalette.accent
    selectedTextColor: AppPalette.bgBase

    background: Rectangle {
        radius: 8
        color: AppPalette.bgBase
        border.color: control.activeFocus ? AppPalette.accent : AppPalette.borderColor
        border.width: control.activeFocus ? 1.5 : 0.5
    }
}
