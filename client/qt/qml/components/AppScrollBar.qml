import QtQuick
import QtQuick.Controls
import "."

ScrollBar {
    id: bar

    width: hovered ? 8 : 4
    policy: ScrollBar.AsNeeded

    opacity: active || hovered ? 1 : 0

    Behavior on opacity {
        NumberAnimation { duration: 150 }
    }
    Behavior on width {
        NumberAnimation { duration: 120 }
    }

    contentItem: Rectangle {
        implicitWidth: 6
        radius: width / 2

        color: bar.pressed ? AppPalette.accent
              : bar.hovered ? AppPalette.accentMid
              : AppPalette.textMuted

        opacity: bar.hovered || bar.pressed ? 0.9 : 0.5
    }

    background: Rectangle {
        implicitWidth: 12
        color: bar.hovered ? AppPalette.bgHover : "transparent"
        radius: 6
    }
}
