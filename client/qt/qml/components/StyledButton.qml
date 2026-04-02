import QtQuick
import QtQuick.Controls
import "."

// Reusable styled button matching the app's design system.
//
// variant:
//   "primary"   — filled accent background, white text; dims when disabled
//   "secondary" — light accent background, accent text; dims when disabled (default)
//   "tertiary"  — transparent, muted border, secondary text; no disabled treatment
//   "ghost"     — transparent with accent highlight on hover, secondary text
//
// labelColor — overrides the computed text color; useful for edge cases
//              (e.g. a "Delete" button that is secondary but uses textSecondary text)
//
// cornerRadius — override the default border radius of 8
//
// All standard Button properties (text, enabled, implicitWidth, implicitHeight,
// font, onClicked, etc.) work as usual.

Button {
    id: control

    property string variant:      "secondary"
    property int    cornerRadius: 8

    // Computed label color based on variant; override at call site if needed.
    property color labelColor: {
        switch (control.variant) {
            case "primary":
                return control.enabled ? AppPalette.bgBase : AppPalette.textMuted
            case "secondary":
                return control.enabled ? AppPalette.accent : AppPalette.textMuted
            default: // "tertiary", "ghost"
                return AppPalette.textSecondary
        }
    }

    font.pixelSize: 13
    implicitHeight: 36

    HoverHandler { cursorShape: Qt.PointingHandCursor }

    background: Rectangle {
        radius: control.cornerRadius
        color: {
            switch (control.variant) {
                case "primary":
                    return control.enabled ? AppPalette.accent : AppPalette.bgSurfaceAlt
                case "secondary":
                    return control.enabled ? AppPalette.accentLight : AppPalette.bgSurfaceAlt
                case "ghost":
                    return control.hovered ? AppPalette.accentLight : "transparent"
                default: // "tertiary"
                    return "transparent"
            }
        }
        border.color: (control.variant === "primary" || control.variant === "secondary")
                      ? (control.enabled ? AppPalette.accent : AppPalette.borderColor)
                      : AppPalette.borderColor
        border.width: 0.5
    }

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.labelColor
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
