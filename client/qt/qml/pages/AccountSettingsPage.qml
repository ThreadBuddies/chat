import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {

    // ── Local ViewModel state ────────────────────────────────────────────────
    property string feedbackText:    ""
    property bool   feedbackIsError: false
    property bool   busy:            false

    onVisibleChanged: {
        if (!visible) {
            feedbackAnim.stop()
            feedbackLabel.opacity = 0
            feedbackText = ""
            newUsernameField.clear()
            oldPasswordField.clear()
            newPasswordField.clear()
            confirmPasswordField.clear()
            busy = false
        }
    }

    function showFeedback(message, isError) {
        feedbackAnim.stop()
        feedbackLabel.opacity = 0
        feedbackText    = message
        feedbackIsError = isError
        feedbackAnim.restart()
    }

    function requestConfirmation(message, action) {
        confirmDialog.message       = message
        confirmDialog.pendingAction = action
        confirmDialog.open()
    }

    // ── Confirmation dialog ───────────────────────────────────────────────────
    Dialog {
        id: confirmDialog
        width: 300
        anchors.centerIn: parent
        modal: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        property string message:       ""
        property var    pendingAction: null

        background: Rectangle {
            radius: 10
            color: AppPalette.bgBase
            border.color: AppPalette.borderColor
            border.width: 1
        }

        contentItem: ColumnLayout {
            spacing: 16
            width: parent.width
            Text {
                text: "Confirm action"
                font.pixelSize: 14
                font.weight: Font.DemiBold
                color: AppPalette.textPrimary
                Layout.fillWidth: true
            }
            Text {
                text: confirmDialog.message
                font.pixelSize: 13
                color: AppPalette.textPrimary
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
                Layout.topMargin: 4
            }

            RowLayout {
                spacing: 8
                Layout.fillWidth: true

                StyledButton {
                    text: "Cancel"
                    variant: "tertiary"
                    Layout.fillWidth: true
                    implicitHeight: 34
                    onClicked: confirmDialog.close()
                }

                StyledButton {
                    text: "Confirm"
                    variant: "primary"
                    Layout.fillWidth: true
                    implicitHeight: 34
                    font.weight: Font.DemiBold
                    onClicked: {
                        if (confirmDialog.pendingAction)
                            confirmDialog.pendingAction()
                        confirmDialog.close()
                    }
                }
            }
        }
    }

    Connections {
        target: appController

        function onChangeUsernameSucceeded() {
            busy = false
            newUsernameField.clear()
            showFeedback("Username changed successfully", false)
        }
        function onChangeUsernameFailed(error) {
            busy = false
            showFeedback(error, true)
        }
        function onChangePasswordSucceeded() {
            busy = false
            oldPasswordField.clear()
            newPasswordField.clear()
            confirmPasswordField.clear()
            showFeedback("Password changed successfully", false)
        }
        function onChangePasswordFailed(error) {
            busy = false
            showFeedback(error, true)
        }
    }

    // ── Background ───────────────────────────────────────────────────────────
    Rectangle {
        anchors.fill: parent
        color: AppPalette.bgBase
    }

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 350)
        spacing: 12

        Text {
            text: "Account Settings"
            font.pixelSize: 20
            font.weight: Font.DemiBold
            color: AppPalette.textPrimary
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            text: "Logged in as: " + appController.currentUsername
            font.pixelSize: 13
            color: AppPalette.textSecondary
            Layout.alignment: Qt.AlignHCenter
        }

        // ── Feedback message ─────────────────────────────────────────────────
        Text {
            id: feedbackLabel
            text: feedbackText
            opacity: 0.0
            visible: opacity > 0
            color: feedbackIsError ? AppPalette.colorError : AppPalette.colorSuccess
            font.pixelSize: 13
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
        }

        SequentialAnimation {
            id: feedbackAnim
            NumberAnimation { target: feedbackLabel; property: "opacity"; to: 1.0; duration: 150 }
            PauseAnimation  { duration: 2500 }
            NumberAnimation { target: feedbackLabel; property: "opacity"; to: 0.0; duration: 500 }
        }

        // ── Change Username ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: AppPalette.borderColor
            Layout.topMargin: 4
        }

        Text {
            text: "Change Username"
            font.pixelSize: 14
            font.weight: Font.DemiBold
            color: AppPalette.textPrimary
        }

        StyledTextField {
            id: newUsernameField
            Layout.fillWidth: true
            placeholderText: "New username"
            unicodeMaxLength: appController.maxUsernameLength
            onAccepted: changeUsernameButton.clicked()
        }

        StyledButton {
            id: changeUsernameButton
            text: "Change Username"
            variant: "secondary"
            Layout.fillWidth: true
            font.weight: Font.DemiBold
            enabled: newUsernameField.text.length > 0 && !busy
            onClicked: {
                var name = newUsernameField.text
                requestConfirmation(
                    "Change your username to \"" + name + "\"?",
                    function() {
                        busy = true
                        appController.changeUsername(name)
                    }
                )
            }
        }

        // ── Change Password ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: AppPalette.borderColor
            Layout.topMargin: 4
        }

        Text {
            text: "Change Password"
            font.pixelSize: 14
            font.weight: Font.DemiBold
            color: AppPalette.textPrimary
        }

        StyledTextField {
            id: oldPasswordField
            Layout.fillWidth: true
            placeholderText: "Current password"
            echoMode: TextInput.Password
            onAccepted: newPasswordField.forceActiveFocus()
        }

        StyledTextField {
            id: newPasswordField
            Layout.fillWidth: true
            placeholderText: "New password"
            echoMode: TextInput.Password
            onAccepted: confirmPasswordField.forceActiveFocus()
        }

        StyledTextField {
            id: confirmPasswordField
            Layout.fillWidth: true
            placeholderText: "Confirm new password"
            echoMode: TextInput.Password
            onAccepted: changePasswordButton.clicked()
        }

        StyledButton {
            id: changePasswordButton
            text: "Change Password"
            variant: "secondary"
            Layout.fillWidth: true
            font.weight: Font.DemiBold
            enabled: oldPasswordField.text.length > 0
                     && newPasswordField.text.length > 0
                     && confirmPasswordField.text.length > 0
                     && !busy
            onClicked: {
                if (newPasswordField.text !== confirmPasswordField.text) {
                    showFeedback("Passwords do not match", true)
                    return
                }
                var oldPwd = oldPasswordField.text
                var newPwd = newPasswordField.text
                requestConfirmation(
                    "Change your password?",
                    function() {
                        busy = true
                        appController.changePassword(oldPwd, newPwd)
                    }
                )
            }
        }

        // ── Back ──────────────────────────────────────────────────────────────
        StyledButton {
            text: "Back"
            variant: "tertiary"
            Layout.alignment: Qt.AlignHCenter
            implicitHeight: 32
            font.pixelSize: 12
            enabled: !busy
            onClicked: appController.goBack()
        }
    }
}
