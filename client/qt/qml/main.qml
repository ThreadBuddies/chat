import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "pages"

ApplicationWindow {
    id: root
    visible: true
    width: 900
    height: 600
    title: "Slightly Pretty Chat"

    StackLayout {
        anchors.fill: parent
        currentIndex: appController.currentPage

        ServerListPage {}   // 0 - ServerList
        ServersPage {}      // 1 - Servers
        AuthPage {}         // 2 - Auth
        ChatPanel {}        // 3 - Chat
    }

    // Error banner at the bottom
    Rectangle {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 10
        width: errorLabel.implicitWidth + 24
        height: errorLabel.implicitHeight + 16
        radius: 6
        color: "#d32f2f"
        visible: appController.errorMessage.length > 0
        opacity: visible ? 1.0 : 0.0

        Behavior on opacity { NumberAnimation { duration: 200 } }

        Label {
            id: errorLabel
            anchors.centerIn: parent
            text: appController.errorMessage
            color: "white"
            font.pixelSize: 13
        }

        MouseArea {
            anchors.fill: parent
            onClicked: appController.clearError()
        }
    }
}
