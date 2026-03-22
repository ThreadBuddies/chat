import QtQuick
import QtQuick.Controls

ScrollBar {
    id: bar

    property color accent
    property color accentMid
    property color textMuted
    property color bgHover

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

        color: bar.pressed ? bar.accent
              : bar.hovered ? bar.accentMid
              : bar.textMuted

        opacity: bar.hovered || bar.pressed ? 0.9 : 0.5
    }

    background: Rectangle {
        implicitWidth: 12
        color: bar.hovered ? bar.bgHover : "transparent"
        radius: 6
    }
}