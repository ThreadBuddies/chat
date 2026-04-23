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
        width: Math.min(parent.width - 40, 350)
        spacing: 12

        Text {
            text: "Authentication"
            font.pixelSize: 20
            font.weight: Font.DemiBold
            color: AppPalette.textPrimary
            Layout.alignment: Qt.AlignHCenter
        }

        StyledTextField {
            id: usernameField
            Layout.fillWidth: true
            placeholderText: "Username"
            unicodeMaxLength: appController.maxUsernameLength
            onAccepted: passwordField.forceActiveFocus()
        }

        StyledTextField {
            id: passwordField
            Layout.fillWidth: true
            placeholderText: "Password"
            echoMode: TextInput.Password
            onAccepted: loginButton.clicked()
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            StyledButton {
                id: loginButton
                text: "Login"
                variant: "primary"
                Layout.fillWidth: true
                Layout.preferredWidth: 0
                font.weight: Font.DemiBold
                enabled: usernameField.text.length > 0 && passwordField.text.length > 0
                onClicked: appController.login(usernameField.text, passwordField.text)
            }

            StyledButton {
                text: "Register"
                variant: "secondary"
                Layout.fillWidth: true
                Layout.preferredWidth: 0
                font.weight: Font.DemiBold
                enabled: usernameField.text.length > 0 && passwordField.text.length > 0
                onClicked: appController.registerUser(usernameField.text, passwordField.text)
            }
        }

        StyledButton {
            text: "Back"
            variant: "tertiary"
            Layout.alignment: Qt.AlignHCenter
            implicitHeight: 32
            font.pixelSize: 12
            onClicked: appController.goBack()
        }
    }
}
