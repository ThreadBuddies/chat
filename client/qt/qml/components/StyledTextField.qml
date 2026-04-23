import QtQuick
import QtQuick.Controls
import App.Validators 1.0
import "."

// Reusable styled text field matching the app's design system.
// Supports all standard TextField properties (placeholderText, echoMode,
// maximumLength, onAccepted, etc.) — set them at the call site.
TextField {
    id: control
    property int unicodeMaxLength: -1
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

    validator: CodePointValidator { maxCodePoints: control.unicodeMaxLength }

    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_Backspace
                && !(event.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier))
                && control.selectionStart === control.selectionEnd) {
            var pos = control.cursorPosition
            if (pos > 0) {
                var newPos = appController.previousGraphemeBoundary(control.text, pos)
                if (newPos >= 0 && newPos < pos - 1) {
                    control.remove(newPos, pos)
                    event.accepted = true
                }
            }
        }
    }
}
