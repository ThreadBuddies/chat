import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 350)
        spacing: 12

        Label {
            text: "Authentication"
            font.pixelSize: 20
            font.bold: true
            Layout.alignment: Qt.AlignHCenter
        }

        TextField {
            id: usernameField
            Layout.fillWidth: true
            placeholderText: "Username"
            onAccepted: passwordField.forceActiveFocus()
        }

        TextField {
            id: passwordField
            Layout.fillWidth: true
            placeholderText: "Password"
            echoMode: TextInput.Password
            onAccepted: loginButton.clicked()
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                id: loginButton
                text: "Login"
                Layout.fillWidth: true
                enabled: usernameField.text.length > 0 && passwordField.text.length > 0
                onClicked: appController.login(usernameField.text, passwordField.text)
            }

            Button {
                text: "Register"
                Layout.fillWidth: true
                enabled: usernameField.text.length > 0 && passwordField.text.length > 0
                onClicked: appController.registerUser(usernameField.text, passwordField.text)
            }
        }

        Button {
            text: "Back"
            Layout.alignment: Qt.AlignHCenter
            onClicked: appController.goBack()
        }
    }
}
