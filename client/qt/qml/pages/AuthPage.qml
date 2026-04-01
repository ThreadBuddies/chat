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

        TextField {
            id: usernameField
            Layout.fillWidth: true
            placeholderText: "Username"
            font.pixelSize: 13
            color: AppPalette.textPrimary
            selectionColor: AppPalette.accent
            selectedTextColor: AppPalette.bgBase
            onAccepted: passwordField.forceActiveFocus()
            background: Rectangle {
                radius: 8
                color: AppPalette.bgBase
                border.color: usernameField.activeFocus ? AppPalette.accent : AppPalette.borderColor
                border.width: usernameField.activeFocus ? 1.5 : 0.5
            }
        }

        TextField {
            id: passwordField
            Layout.fillWidth: true
            placeholderText: "Password"
            font.pixelSize: 13
            color: AppPalette.textPrimary
            selectionColor: AppPalette.accent
            selectedTextColor: AppPalette.bgBase
            echoMode: TextInput.Password
            onAccepted: loginButton.clicked()
            background: Rectangle {
                radius: 8
                color: AppPalette.bgBase
                border.color: passwordField.activeFocus ? AppPalette.accent : AppPalette.borderColor
                border.width: passwordField.activeFocus ? 1.5 : 0.5
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                id: loginButton
                text: "Login"
                Layout.fillWidth: true
                Layout.preferredWidth: 0
                implicitHeight: 36
                font.pixelSize: 13
                font.weight: Font.DemiBold
                enabled: usernameField.text.length > 0 && passwordField.text.length > 0
                onClicked: appController.login(usernameField.text, passwordField.text)
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
                text: "Register"
                Layout.fillWidth: true
                Layout.preferredWidth: 0
                implicitHeight: 36
                font.pixelSize: 13
                font.weight: Font.DemiBold
                enabled: usernameField.text.length > 0 && passwordField.text.length > 0
                onClicked: appController.registerUser(usernameField.text, passwordField.text)
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

        Button {
            text: "Back"
            Layout.alignment: Qt.AlignHCenter
            implicitHeight: 32
            font.pixelSize: 12
            onClicked: appController.goBack()
            background: Rectangle {
                radius: 8
                color: "transparent"
                border.color: AppPalette.borderColor
                border.width: 0.5
            }
            contentItem: Text {
                text: parent.text; font: parent.font
                color: AppPalette.textSecondary
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            HoverHandler { cursorShape: Qt.PointingHandCursor }
        }
    }
}
