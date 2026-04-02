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
            text: "Available Servers"
            font.pixelSize: 20
            font.weight: Font.DemiBold
            color: AppPalette.textPrimary
            Layout.alignment: Qt.AlignHCenter
        }

        // ── Server list ───────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 300
            color: AppPalette.bgBase
            border.color: AppPalette.borderColor
            border.width: 1
            radius: 8

            ListView {
                id: serversView
                anchors.fill: parent
                anchors.margins: 4
                model: appController.serverListModel
                clip: true
                currentIndex: -1

                delegate: ItemDelegate {
                    width: serversView.width
                    highlighted: serversView.currentIndex === index
                    onClicked: serversView.currentIndex = index

                    background: Rectangle {
                        radius: 6
                        color: parent.highlighted ? AppPalette.bgSelected
                             : parent.hovered     ? AppPalette.bgHover
                             : "transparent"
                    }
                    contentItem: Text {
                        text: model.host
                        font.pixelSize: 13
                        color: AppPalette.textPrimary
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: 8
                    }
                    HoverHandler { cursorShape: Qt.PointingHandCursor }
                }

                Text {
                    anchors.centerIn: parent
                    text: "No servers available"
                    color: AppPalette.textMuted
                    font.pixelSize: 13
                    visible: serversView.count === 0
                }
            }
        }

        // ── Actions ───────────────────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            StyledButton {
                text: "Connect"
                variant: "primary"
                enabled: serversView.currentIndex >= 0
                Layout.fillWidth: true
                font.weight: Font.DemiBold
                onClicked: {
                    var host = appController.serverListModel.data(
                        appController.serverListModel.index(serversView.currentIndex, 0),
                        Qt.UserRole + 1 // HostRole
                    )
                    appController.connectToServer(host)
                }
            }

            StyledButton {
                text: "Back"
                variant: "tertiary"
                onClicked: appController.goBack()
            }
        }
    }
}
