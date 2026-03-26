import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 500)
        spacing: 12

        Label {
            text: "Available Servers"
            font.pixelSize: 20
            font.bold: true
            Layout.alignment: Qt.AlignHCenter
        }

        Rectangle {
            Layout.fillWidth: true
            height: 300
            border.color: "#ccc"
            border.width: 1
            radius: 4

            ListView {
                id: serversView
                anchors.fill: parent
                anchors.margins: 2
                model: appController.serverListModel
                clip: true
                currentIndex: -1

                delegate: ItemDelegate {
                    width: serversView.width
                    text: model.host
                    highlighted: serversView.currentIndex === index
                    onClicked: serversView.currentIndex = index
                }

                Label {
                    anchors.centerIn: parent
                    text: "No servers available"
                    color: "#999"
                    visible: serversView.count === 0
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: "Connect"
                enabled: serversView.currentIndex >= 0
                Layout.fillWidth: true
                onClicked: {
                    var host = appController.serverListModel.data(
                        appController.serverListModel.index(serversView.currentIndex, 0),
                        Qt.UserRole + 1 // HostRole
                    )
                    appController.connectToServer(host)
                }
            }

            Button {
                text: "Back"
                onClicked: appController.goBack()
            }
        }
    }
}
