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

                Button {
                    text: "Cancel"
                    Layout.fillWidth: true
                    implicitHeight: 34
                    font.pixelSize: 13
                    onClicked: confirmDialog.close()
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

                Button {
                    text: "Confirm"
                    Layout.fillWidth: true
                    implicitHeight: 34
                    font.pixelSize: 13
                    font.weight: Font.DemiBold
                    onClicked: {
                        if (confirmDialog.pendingAction)
                            confirmDialog.pendingAction()
                        confirmDialog.close()
                    }
                    background: Rectangle {
                        radius: 8
                        color: AppPalette.accent
                        border.color: AppPalette.accent
                        border.width: 0.5
                    }
                    contentItem: Text {
                        text: parent.text; font: parent.font
                        color: AppPalette.bgBase
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    HoverHandler { cursorShape: Qt.PointingHandCursor }
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

        TextField {
            id: newUsernameField
            Layout.fillWidth: true
            placeholderText: "New username"
            font.pixelSize: 13
            color: AppPalette.textPrimary
            selectionColor: AppPalette.accent
            selectedTextColor: AppPalette.bgBase
            onAccepted: changeUsernameButton.clicked()
            background: Rectangle {
                radius: 8
                color: AppPalette.bgBase
                border.color: newUsernameField.activeFocus ? AppPalette.accent : AppPalette.borderColor
                border.width: newUsernameField.activeFocus ? 1.5 : 0.5
            }
        }

        Button {
            id: changeUsernameButton
            text: "Change Username"
            Layout.fillWidth: true
            implicitHeight: 36
            font.pixelSize: 13
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

        TextField {
            id: oldPasswordField
            Layout.fillWidth: true
            placeholderText: "Current password"
            font.pixelSize: 13
            color: AppPalette.textPrimary
            selectionColor: AppPalette.accent
            selectedTextColor: AppPalette.bgBase
            echoMode: TextInput.Password
            onAccepted: newPasswordField.forceActiveFocus()
            background: Rectangle {
                radius: 8
                color: AppPalette.bgBase
                border.color: oldPasswordField.activeFocus ? AppPalette.accent : AppPalette.borderColor
                border.width: oldPasswordField.activeFocus ? 1.5 : 0.5
            }
        }

        TextField {
            id: newPasswordField
            Layout.fillWidth: true
            placeholderText: "New password"
            font.pixelSize: 13
            color: AppPalette.textPrimary
            selectionColor: AppPalette.accent
            selectedTextColor: AppPalette.bgBase
            echoMode: TextInput.Password
            onAccepted: confirmPasswordField.forceActiveFocus()
            background: Rectangle {
                radius: 8
                color: AppPalette.bgBase
                border.color: newPasswordField.activeFocus ? AppPalette.accent : AppPalette.borderColor
                border.width: newPasswordField.activeFocus ? 1.5 : 0.5
            }
        }

        TextField {
            id: confirmPasswordField
            Layout.fillWidth: true
            placeholderText: "Confirm new password"
            font.pixelSize: 13
            color: AppPalette.textPrimary
            selectionColor: AppPalette.accent
            selectedTextColor: AppPalette.bgBase
            echoMode: TextInput.Password
            onAccepted: changePasswordButton.clicked()
            background: Rectangle {
                radius: 8
                color: AppPalette.bgBase
                border.color: confirmPasswordField.activeFocus ? AppPalette.accent : AppPalette.borderColor
                border.width: confirmPasswordField.activeFocus ? 1.5 : 0.5
            }
        }

        Button {
            id: changePasswordButton
            text: "Change Password"
            Layout.fillWidth: true
            implicitHeight: 36
            font.pixelSize: 13
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

        // ── Back ──────────────────────────────────────────────────────────────
        Button {
            text: "Back"
            Layout.alignment: Qt.AlignHCenter
            implicitHeight: 32
            font.pixelSize: 12
            enabled: !busy
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
