import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    Rectangle {
        anchors.fill: parent
        color: AppPalette.bgBase
    }

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 500)
        spacing: 12

        Text {
            text: "Saved Servers"
            font.pixelSize: 20
            font.weight: Font.DemiBold
            color: AppPalette.textPrimary
            Layout.alignment: Qt.AlignHCenter
        }

        // ── Server list ───────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 250
            color: AppPalette.bgBase
            border.color: AppPalette.borderColor
            border.width: 1
            radius: 8

            ListView {
                id: serverListView
                anchors.fill: parent
                anchors.margins: 4
                model: configService.servers
                clip: true
                currentIndex: -1

                delegate: ItemDelegate {
                    width: serverListView.width
                    highlighted: serverListView.currentIndex === index
                    onClicked: serverListView.currentIndex = index

                    background: Rectangle {
                        radius: 6
                        color: parent.highlighted ? AppPalette.bgSelected
                             : parent.hovered     ? AppPalette.bgHover
                             : "transparent"
                    }
                    contentItem: Text {
                        text: modelData
                        font.pixelSize: 13
                        color: AppPalette.textPrimary
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: 8
                    }
                    HoverHandler { cursorShape: Qt.PointingHandCursor }
                }

                Text {
                    anchors.centerIn: parent
                    text: "No servers added yet"
                    color: AppPalette.textMuted
                    font.pixelSize: 13
                    visible: serverListView.count === 0
                }
            }
        }

        // ── Add server ────────────────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: serverInput
                Layout.fillWidth: true
                placeholderText: "ws://localhost:8848"
                font.pixelSize: 13
                color: AppPalette.textPrimary
                selectionColor: AppPalette.accent
                selectedTextColor: AppPalette.bgBase
                onAccepted: addButton.clicked()
                background: Rectangle {
                    radius: 8
                    color: AppPalette.bgBase
                    border.color: serverInput.activeFocus ? AppPalette.accent : AppPalette.borderColor
                    border.width: serverInput.activeFocus ? 1.5 : 0.5
                }
            }

            Button {
                id: addButton
                text: "Add"
                Layout.fillHeight: true
                implicitWidth: 72
                font.pixelSize: 13
                font.weight: Font.DemiBold
                enabled: serverInput.text.trim().length > 0
                onClicked: {
                    if (serverInput.text.trim().length > 0) {
                        configService.addServer(serverInput.text.trim())
                        serverInput.clear()
                    }
                }
                background: Rectangle {
                    radius: 8
                    color: parent.enabled ? AppPalette.accentLight : AppPalette.bgSurfaceAlt
                    border.color: parent.enabled ? AppPalette.accent : AppPalette.borderColor
                    border.width: 0.5
                }
                contentItem: Text {
                    text: parent.text; font: parent.font
                    color: parent.enabled ? AppPalette.accent : AppPalette.textMuted
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                HoverHandler { cursorShape: Qt.PointingHandCursor }
            }
        }

        // ── Actions ───────────────────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: "Connect"
                enabled: serverListView.currentIndex >= 0
                Layout.fillWidth: true
                implicitHeight: 36
                font.pixelSize: 13
                font.weight: Font.DemiBold
                onClicked: {
                    var url = configService.servers[serverListView.currentIndex]
                    appController.connectToAggregator(url)
                }
                background: Rectangle {
                    radius: 8
                    color: parent.enabled ? AppPalette.accent : AppPalette.bgSurfaceAlt
                    border.color: parent.enabled ? AppPalette.accent : AppPalette.borderColor
                    border.width: 0.5
                }
                contentItem: Text {
                    text: parent.text; font: parent.font
                    color: parent.enabled ? AppPalette.bgBase : AppPalette.textMuted
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                HoverHandler { cursorShape: Qt.PointingHandCursor }
            }

            Button {
                text: "Delete"
                enabled: serverListView.currentIndex >= 0
                implicitHeight: 36
                implicitWidth: 72
                font.pixelSize: 13
                onClicked: {
                    var url = configService.servers[serverListView.currentIndex]
                    configService.removeServer(url)
                    serverListView.currentIndex = -1
                }
                background: Rectangle {
                    radius: 8
                    color: parent.enabled ? AppPalette.accentLight : AppPalette.bgSurfaceAlt
                    border.color: parent.enabled ? AppPalette.accent : AppPalette.borderColor
                    border.width: 0.5
                }
                contentItem: Text {
                    text: parent.text; font: parent.font
                    color: parent.enabled ? AppPalette.textSecondary : AppPalette.textMuted
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                HoverHandler { cursorShape: Qt.PointingHandCursor }
            }
        }
    }
}
