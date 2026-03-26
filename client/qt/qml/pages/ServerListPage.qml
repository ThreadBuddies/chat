import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 500)
        spacing: 12

        Label {
            text: "Saved Servers"
            font.pixelSize: 20
            font.bold: true
            Layout.alignment: Qt.AlignHCenter
        }

        Rectangle {
            Layout.fillWidth: true
            height: 250
            border.color: "#ccc"
            border.width: 1
            radius: 4

            ListView {
                id: serverListView
                anchors.fill: parent
                anchors.margins: 2
                model: configService.servers
                clip: true
                currentIndex: -1

                delegate: ItemDelegate {
                    width: serverListView.width
                    text: modelData
                    highlighted: serverListView.currentIndex === index
                    onClicked: serverListView.currentIndex = index
                }

                Label {
                    anchors.centerIn: parent
                    text: "No servers added yet"
                    color: "#999"
                    visible: serverListView.count === 0
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: serverInput
                Layout.fillWidth: true
                placeholderText: "ws://localhost:8848"
                onAccepted: addButton.clicked()
            }

            Button {
                id: addButton
                text: "Add"
                onClicked: {
                    if (serverInput.text.trim().length > 0) {
                        configService.addServer(serverInput.text.trim())
                        serverInput.clear()
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: "Connect"
                enabled: serverListView.currentIndex >= 0
                Layout.fillWidth: true
                onClicked: {
                    var url = configService.servers[serverListView.currentIndex]
                    appController.connectToAggregator(url)
                }
            }

            Button {
                text: "Delete"
                enabled: serverListView.currentIndex >= 0
                onClicked: {
                    var url = configService.servers[serverListView.currentIndex]
                    configService.removeServer(url)
                    serverListView.currentIndex = -1
                }
            }
        }
    }
}
