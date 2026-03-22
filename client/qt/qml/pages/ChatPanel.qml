import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "../components"

Item {
    id: root

    // ── Palette ──────────────────────────────────────────────
    // Neutral tones inspired by the mockup; easy to swap.
    readonly property color bgBase:       "#ffffff"
    readonly property color bgSurface:    "#f7f7f8"
    readonly property color bgSurfaceAlt: "#f0f0f2"
    readonly property color bgHover:      "#eaeaed"
    readonly property color bgSelected:   "#e0ecfa"
    readonly property color accent:       "#2563EB"   // blue-600
    readonly property color accentLight:  "#eff6ff"   // blue-50
    readonly property color accentMid:    "#93c5fd"   // blue-300
    readonly property color textPrimary:  "#18181b"
    readonly property color textSecondary:"#71717a"
    readonly property color textMuted:    "#a1a1aa"
    readonly property color borderColor:  "#e4e4e7"
    readonly property color greenOnline:  "#16a34a"
    readonly property color orangeWarn:   "#d97706"
    readonly property color coralUser:    "#D85A30"

    // ── Helpers ──────────────────────────────────────────────
    function rightsLabel(rights) {
        if (rights === 2) return "owner"
        if (rights === 1) return "mod"
        if (rights === 3) return "admin"
        return ""
    }
    function rightsColor(rights) {
        if (rights === 2) return orangeWarn
        if (rights === 1) return accent
        if (rights === 3) return "#dc2626"
        return "transparent"
    }
    // Stable per-user color from username hash
    function userColor(name) {
        var colors = ["#2563EB","#D85A30","#7c3aed","#0d9488","#c026d3","#059669","#e11d48","#ca8a04"]
        var h = 0
        for (var i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0
        return colors[Math.abs(h) % colors.length]
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ═══════════════════════════════════════════════════
        //  LEFT SIDEBAR
        // ═══════════════════════════════════════════════════
        Rectangle {
            Layout.preferredWidth: 220
            Layout.fillHeight: true
            color: bgSurface
            border.color: borderColor
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                // ── Tab bar ──────────────────────────────
                Row {
                    Layout.fillWidth: true
                    height: 40

                    Repeater {
                        model: ["My rooms", "Browse"]
                        delegate: Item {
                            width: parent.width / 2
                            height: 40

                            property bool active: tabBar.currentIndex === index

                            Rectangle {
                                anchors.fill: parent
                                color: parent.active ? "transparent" : "transparent"

                                Text {
                                    anchors.centerIn: parent
                                    text: modelData
                                    font.pixelSize: 12
                                    font.weight: parent.parent.active ? Font.DemiBold : Font.Normal
                                    color: parent.parent.active ? accent : textSecondary
                                }
                                Rectangle {
                                    anchors.bottom: parent.bottom
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    width: parent.width - 24
                                    height: 2
                                    color: accent
                                    visible: parent.parent.active
                                    radius: 1
                                }
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: tabBar.currentIndex = index
                            }
                        }
                    }
                    // invisible helper to hold state
                    QtObject {
                        id: tabBar
                        property int currentIndex: 0
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: borderColor }

                // ── Room lists ────────────────────────────
                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: tabBar.currentIndex

                    // ── Tab 0: My rooms ──────────────────
                    ListView {
                        id: myRoomsView
                        clip: true
                        model: appController.roomListModel
                        
                        ScrollBar.vertical: AppScrollBar {
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.rightMargin: 2

                            accent: root.accent
                            accentMid: root.accentMid
                            textMuted: root.textMuted
                            bgHover: root.bgHover
                        }

                        delegate: Item {
                            visible: model.isJoined
                            width: myRoomsView.width
                            height: visible ? implicitHeight : 0

                            implicitHeight: roomMyCol.implicitHeight + 12

                            Rectangle {
                                id: myRoomBg
                                anchors.fill: parent
                                anchors.margins: 3
                                radius: 8
                                color: {
                                    var selected = model.roomId === appController.currentRoomId
                                    if (selected) return bgSelected
                                    if (myRoomMa.containsMouse) return bgHover
                                    return "transparent"
                                }
                                border.color: "transparent"
                                border.width: 0

                                ColumnLayout {
                                    id: roomMyCol
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.margins: 10
                                    spacing: 2

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 6

                                        Text {
                                            text: model.roomName
                                            font.pixelSize: 13
                                            font.weight: Font.DemiBold
                                            color: textPrimary
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                        }

                                        // Online indicator for current room
                                        Rectangle {
                                            width: 7; height: 7; radius: 3.5
                                            color: greenOnline
                                            visible: model.roomId === appController.currentRoomId
                                        }
                                    }
                                }

                                MouseArea {
                                    id: myRoomMa
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    hoverEnabled: true
                                    onClicked: appController.joinRoom(model.roomId)
                                }
                            }
                        }

                        // Empty state
                        Text {
                            anchors.centerIn: parent
                            text: "No rooms yet"
                            color: textMuted
                            font.pixelSize: 13
                            visible: myRoomsView.count === 0
                        }
                    }

                    // ── Tab 1: Browse (public rooms) ─────
                    ListView {
                        id: browseView
                        clip: true
                        model: appController.roomListModel

                        ScrollBar.vertical: AppScrollBar {
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.rightMargin: 2

                            accent: root.accent
                            accentMid: root.accentMid
                            textMuted: root.textMuted
                            bgHover: root.bgHover
                        }
                        delegate: Item {
                            visible: !model.isJoined
                            width: browseView.width
                            height: visible ? 48 : 0

                            Rectangle {
                                id: browseBg
                                anchors.fill: parent
                                anchors.margins: 3
                                radius: 8
                                color: "transparent"

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    anchors.rightMargin: 10
                                    spacing: 8

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 0
                                        Text {
                                            text: model.roomName
                                            font.pixelSize: 13
                                            font.weight: Font.DemiBold
                                            color: textPrimary
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                        }
                                    }

                                    // Join button — only for public rooms
                                    Button {
                                        text: "Join"
                                        implicitHeight: 26
                                        implicitWidth: 52
                                        font.pixelSize: 11
                                        font.weight: Font.DemiBold
                                        onClicked: appController.becomeMember(model.roomId)
                                        background: Rectangle {
                                            radius: 6
                                            color: accentLight
                                            border.color: accent
                                            border.width: 0.5
                                        }
                                        contentItem: Text {
                                            text: parent.text
                                            font: parent.font
                                            color: accent
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter
                                        }
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    // no click — use the Join button
                                    onEntered: browseBg.color = bgHover
                                    onExited:  browseBg.color = "transparent"
                                    // let button clicks pass through
                                    propagateComposedEvents: true
                                    onClicked: function(mouse) { mouse.accepted = false }
                                    onPressed: function(mouse) { mouse.accepted = false }
                                    onReleased: function(mouse) { mouse.accepted = false }
                                }
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            text: "No public rooms"
                            color: textMuted
                            font.pixelSize: 13
                            visible: browseView.count === 0
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: borderColor }

                // ── Bottom controls ──────────────────────
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.margins: 8
                    spacing: 4

                    // Create room
                    Button {
                        id: createRoomBtn
                        text: "+ Create room"
                        Layout.fillWidth: true
                        implicitHeight: 34
                        font.pixelSize: 12
                        onClicked: createRoomDialog.open()
                        background: Rectangle {
                            radius: 8
                            color: bgBase
                            border.color: borderColor
                            border.width: 0.5
                        }
                        contentItem: Text {
                            text: parent.text
                            font: parent.font
                            color: textPrimary
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4

                        Button {
                            text: "Account"
                            Layout.fillWidth: true
                            implicitHeight: 30
                            font.pixelSize: 11
                            background: Rectangle { radius: 8; color: parent.hovered ? accentLight : "transparent"; border.color: borderColor; border.width: 0.5 }
                            contentItem: Text {
                                text: parent.text; font: parent.font; color: textSecondary
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                            }
                            onClicked: {
                                // TODO: account settings
                            }
                            HoverHandler {
                                cursorShape: Qt.PointingHandCursor
                            }
                        }
                        Button {
                            text: "Logout"
                            Layout.fillWidth: true
                            implicitHeight: 30
                            font.pixelSize: 11
                            background: Rectangle { radius: 8; color: parent.hovered ? accentLight : "transparent"; border.color: borderColor; border.width: 0.5 }
                            contentItem: Text {
                                text: parent.text; font: parent.font; color: textSecondary
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                            }
                            onClicked: appController.logout()
                            HoverHandler {
                                cursorShape: Qt.PointingHandCursor
                            }
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════
        //  CENTER — CHAT
        // ═══════════════════════════════════════════════════
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: appController.currentRoomId >= 0 ? 1 : 0

            // ── Page 0: no room selected ─────────────────
            Item {
                Text {
                    anchors.centerIn: parent
                    text: "Select a room to start chatting"
                    font.pixelSize: 16
                    color: textMuted
                }
            }

            // ── Page 1: active chat ──────────────────────
            ColumnLayout {
                spacing: 0

                // ── Room header ──────────────────────────
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 48
                    color: bgBase
                    border.color: borderColor
                    border.width: 1

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 16
                        anchors.rightMargin: 16
                        spacing: 8

                        Text {
                            text: appController.currentRoomName
                            font.pixelSize: 15
                            font.weight: Font.DemiBold
                            color: textPrimary
                            Layout.fillWidth: true
                        }
                        Text {
                            visible: appController.onlineCount !== undefined
                            text: (appController.onlineCount || 0) + " online"
                            font.pixelSize: 11
                            color: textMuted
                        }
                        Button {
                            text: "Members"
                            implicitHeight: 28
                            font.pixelSize: 11
                            onClicked: rightSidebar.visible = !rightSidebar.visible
                            background: Rectangle {
                                radius: 6; color: "transparent"
                                border.color: borderColor; border.width: 0.5
                            }
                            contentItem: Text {
                                text: parent.text; font: parent.font; color: textSecondary
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                            }
                        }
                    }
                }

                // ── Messages ─────────────────────────────
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    ListView {
                        id: messageView
                        anchors.fill: parent
                        model: appController.messageListModel
                        clip: true
                        verticalLayoutDirection: ListView.BottomToTop
                        spacing: 2

                        ScrollBar.vertical: AppScrollBar {
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.rightMargin: 2

                            accent: root.accent
                            accentMid: root.accentMid
                            textMuted: root.textMuted
                            bgHover: root.bgHover
                        }

                        delegate: Item {
                            width: messageView.width
                            height: msgCol.implicitHeight + 14

                            Rectangle {
                                anchors.fill: parent
                                color: index % 2 === 0 ? bgBase : bgSurface
                            }

                            ColumnLayout {
                                id: msgCol
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.margins: 16
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: 2

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Text {
                                        text: model.username
                                        font.pixelSize: 13
                                        font.weight: Font.DemiBold
                                        color: userColor(model.username)
                                    }
                                    Item { Layout.fillWidth: true }
                                    Text {
                                        text: model.formattedTime
                                        font.pixelSize: 10
                                        color: textMuted
                                    }
                                }

                                Text {
                                    text: model.messageText
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                    font.pixelSize: 14
                                    color: textPrimary
                                    lineHeight: 1.4
                                }
                            }
                        }

                        onAtYBeginningChanged: {
                            if (atYBeginning && count > 0)
                                appController.loadOlderMessages()
                        }

                        onAtYEndChanged: {
                            if (atYEnd && count > 0)
                                appController.loadNewerMessages()
                        }

                        Text {
                            anchors.centerIn: parent
                            text: "No messages yet — say hi!"
                            color: textMuted
                            font.pixelSize: 14
                            visible: messageView.count === 0
                        }
                    }

                    // ── Scroll-to-bottom FAB ────────────────
                    Rectangle {
                        id: scrollToBottomBtn
                        width: 36; height: 36; radius: 18
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        anchors.rightMargin: 16
                        anchors.bottomMargin: 12
                        visible: opacity > 0
                        opacity: !messageView.atYEnd ? 1.0 : 0.0
                        color: scrollBtnMa.containsMouse ? bgHover : bgBase
                        border.color: borderColor
                        border.width: 1

                        Behavior on opacity {
                            NumberAnimation { duration: 200; easing.type: Easing.InOutQuad }
                        }

                        // Down-chevron drawn with a rotated rectangle
                        Text {
                            anchors.centerIn: parent
                            text: "\u2193"          // ↓
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                            color: accent
                        }

                        layer.enabled: true
                        layer.effect: MultiEffect {
                            shadowEnabled: true
                            shadowColor: "#22000000"
                            shadowBlur: 0.3
                            shadowVerticalOffset: 2
                            shadowHorizontalOffset: 0
                        }

                        MouseArea {
                            id: scrollBtnMa
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            hoverEnabled: true
                            onClicked: {
                                if (appController.messageListModel.canLoadNewer)
                                    appController.jumpToLatest()
                                else
                                    messageView.positionViewAtBeginning()
                            }
                        }
                    }
                }

                // ── Typing indicator ─────────────────────
                Item {
                    Layout.fillWidth: true
                    implicitHeight: typingLabel.visible ? 22 : 0
                    Layout.leftMargin: 16

                    Text {
                        id: typingLabel
                        visible: appController.typingUsers !== undefined && appController.typingUsers.length > 0
                        text: appController.typingUsers || ""
                        font.pixelSize: 12
                        font.italic: true
                        color: textMuted
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }

                // ── Message input ────────────────────────
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 54
                    color: bgSurface
                    border.color: borderColor
                    border.width: 1

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 8

                        TextField {
                            id: msgInput
                            Layout.fillWidth: true
                            placeholderText: "Type a message..."
                            font.pixelSize: 13

                            background: Rectangle {
                                radius: 8
                                color: bgBase
                                border.color: msgInput.activeFocus ? accent : borderColor
                                border.width: msgInput.activeFocus ? 1.5 : 0.5
                            }

                            onTextChanged: {
                                if (text.length > 0)
                                    appController.startTyping()
                            }

                            onAccepted: {
                                if (text.trim().length > 0) {
                                    appController.stopTyping()
                                    appController.sendMessage(text)
                                    clear()
                                }
                            }

                            Keys.onEscapePressed: {
                                appController.stopTyping()
                                clear()
                            }
                        }

                        Button {
                            text: "Send"
                            enabled: msgInput.text.trim().length > 0
                            implicitHeight: 36
                            implicitWidth: 64
                            font.pixelSize: 12
                            font.weight: Font.DemiBold
                            onClicked: {
                                appController.stopTyping()
                                appController.sendMessage(msgInput.text)
                                msgInput.clear()
                            }
                            background: Rectangle {
                                radius: 8
                                color: parent.enabled ? accentLight : bgSurfaceAlt
                                border.color: parent.enabled ? accent : borderColor
                                border.width: 0.5
                            }
                            contentItem: Text {
                                text: parent.text; font: parent.font
                                color: parent.enabled ? accent : textMuted
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════
        //  RIGHT SIDEBAR — MEMBERS
        // ═══════════════════════════════════════════════════
        Rectangle {
            id: rightSidebar
            Layout.preferredWidth: 170
            Layout.fillHeight: true
            color: bgSurface
            border.color: borderColor
            border.width: 1
            visible: appController.currentRoomId >= 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 4

                ListView {
                    id: membersView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: appController.userListModel
                    clip: true
                    spacing: 2

                    delegate: Item {
                        // Show all members (online first ideally — depends on model sorting)
                        width: membersView.width
                        height: 24

                        RowLayout {
                            anchors.fill: parent
                            spacing: 6

                            Rectangle {
                                width: 7; height: 7; radius: 3.5
                                color: model.count > 0 ? greenOnline : borderColor
                            }
                            Text {
                                text: model.userName
                                font.pixelSize: 12
                                color: model.count > 0 ? textPrimary : textMuted
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                            Text {
                                visible: rightsLabel(model.userRights).length > 0
                                text: rightsLabel(model.userRights)
                                font.pixelSize: 10
                                color: rightsColor(model.userRights)
                            }
                        }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    //  CREATE ROOM DIALOG
    // ═══════════════════════════════════════════════════════
    Dialog {
        id: createRoomDialog
        title: "Create room"
        anchors.centerIn: parent
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 300

        onAccepted: {
            var name = newRoomField.text.trim()
            if (name.length > 0) {
                appController.createRoom(name)
                newRoomField.clear()
            }
        }
        onRejected: newRoomField.clear()

        ColumnLayout {
            anchors.fill: parent
            spacing: 8

            Text {
                text: "Room name"
                font.pixelSize: 13
                color: textSecondary
            }
            TextField {
                id: newRoomField
                Layout.fillWidth: true
                placeholderText: "e.g. my-room"
                font.pixelSize: 13
                maximumLength: 64
                onAccepted: createRoomDialog.accept()

                background: Rectangle {
                    radius: 8
                    color: bgBase
                    border.color: newRoomField.activeFocus ? accent : borderColor
                    border.width: newRoomField.activeFocus ? 1.5 : 0.5
                }
            }
        }
    }
}
