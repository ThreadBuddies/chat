import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "pages"
import "components"

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

        width: Math.min(parent.width * 0.7, errorLabel.implicitWidth + 32)
        height: errorLabel.implicitHeight + 20

        radius: 8
        color: AppPalette.colorErrorBg
        border.color: AppPalette.colorErrorBorder
        border.width: 1
        
        visible: appController.errorMessage.length > 0
        opacity: visible ? 1.0 : 0.0

        Behavior on opacity { NumberAnimation { duration: 200 } }

        Label {
            id: errorLabel
            anchors.centerIn: parent
            text: appController.errorMessage
            color: AppPalette.colorError
            font.pixelSize: 13
        }

        MouseArea {
            anchors.fill: parent
            onClicked: appController.clearError()
        }
    }
}
