import QtQuick
import QtQuick.Controls
import App.Validators 1.0
import "."

// Multi-line styled text area. Enter emits accepted(); Shift+Enter inserts a
// newline. Escape emits escaped(). Grows vertically with content up to
// maxLines lines, then scrolls. Over-limit input is reverted at the edit
ScrollView {
    id: control

    property int unicodeMaxLength: -1
    property alias text: textArea.text
    property alias placeholderText: textArea.placeholderText
    property int maxLines: 6

    signal accepted()
    signal escaped()

    function clear()            { textArea.clear() }
    function forceActiveFocus() { textArea.forceActiveFocus() }

    // Grow with content up to maxLines, then scroll.
    implicitHeight: {
        var pad  = textArea.topPadding + textArea.bottomPadding
        var line = textArea.font.pixelSize * 1.4
        var cap  = line * maxLines + pad
        return Math.min(textArea.implicitHeight, cap)
    }

    ScrollBar.vertical: AppScrollBar { }

    CodePointValidator {
        id: cpv
        maxCodePoints: control.unicodeMaxLength
    }

    TextArea {
        id: textArea
        font.pixelSize: 13
        color: AppPalette.textPrimary
        selectionColor: AppPalette.accent
        selectedTextColor: AppPalette.bgBase
        wrapMode: TextArea.Wrap

        background: Rectangle {
            radius: 8
            color: AppPalette.bgBase
            border.color: textArea.activeFocus ? AppPalette.accent : AppPalette.borderColor
            border.width: textArea.activeFocus ? 1.5 : 0.5
        }

        property string _lastValidText: ""
        property bool   _reverting: false
        onTextChanged: {
            if (_reverting) return
            if (cpv.isOverLimit(text)) {
                _reverting = true
                var pos = cursorPosition
                text = _lastValidText
                cursorPosition = Math.min(Math.max(0, pos - 1), text.length)
                _reverting = false
                return
            }
            _lastValidText = text
        }

        Keys.onPressed: function(event) {
            if (event.key === Qt.Key_Escape) {
                event.accepted = true
                control.escaped()
                return
            }
            // Enter alone → accepted; Shift+Enter inserts a newline (default).
            if ((event.key === Qt.Key_Return || event.key === Qt.Key_Enter)
                    && !(event.modifiers & Qt.ShiftModifier)) {
                event.accepted = true
                control.accepted()
                return
            }
            // ZWJ-aware backspace.
            if (event.key === Qt.Key_Backspace
                    && !(event.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier))
                    && textArea.selectionStart === textArea.selectionEnd) {
                var pos = textArea.cursorPosition
                if (pos > 0) {
                    var newPos = appController.previousGraphemeBoundary(textArea.text, pos)
                    if (newPos >= 0 && newPos < pos - 1) {
                        textArea.remove(newPos, pos)
                        event.accepted = true
                    }
                }
            }
        }
    }
}
