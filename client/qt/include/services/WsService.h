#pragma once

#include <QObject>
#include <QWebSocket>
#include <QString>
#include <QStringList>
#include <QList>

#include <models/RoomListModel.h>
#include <models/MessageListModel.h>
#include <models/User.h>

namespace chat {
class Envelope;
}

namespace qt_client {

class WebSocketService : public QObject {
    Q_OBJECT

public:
    explicit WebSocketService(QObject* parent = nullptr);
    ~WebSocketService() override;

    Q_INVOKABLE void connectToUrl(const QString& url);
    Q_INVOKABLE void disconnect();

    // --- Send methods ---
    void sendGetServers();
    void sendInitialAuth(const QString& username);
    void sendAuth(std::string&& hash, std::optional<std::string>&& password = std::nullopt, std::optional<std::string>&& salt = std::nullopt);
    void sendInitialRegister(const QString& username);
    void sendRegister(std::string&& salt, std::string&& hash);
    void sendJoinRoom(int32_t roomId);
    void sendCreateRoom(const QString& name);
    void sendMessage(const QString& text);
    void sendGetMessages(int32_t limit, qint64 offsetTs);
    void sendLogout();
    void sendTypingStart();
    void sendTypingStop();
    void sendBecomeMember(int32_t roomId);
    void sendChangeUsername(const QString& newUsername);
    void sendGetMySalt();
    void sendChangePassword(std::string oldHash, std::string newHash, std::string newSalt);
    void sendAssignRole(int32_t roomId, int32_t userId, chat::UserRights role);

signals:
    // Connection lifecycle
    void connected();
    void disconnected();
    void error(const QString& message);

    // Server hello
    void serverHelloReceived(bool isAggregator);

    // Auth
    void initialAuthResponse(bool success, const QString& salt);
    void authSuccess(User user, const QList<RoomData>& rooms);
    void authFailure(const QString& error);

    // Registration
    void initialRegisterResponse(bool success, const QString& error);
    void registerSuccess();
    void registerFailure(const QString& error);

    // Server discovery
    void serversReceived(const QStringList& servers);
    void serverAdded(const QString& host);
    void serverRemoved(const QString& host);

    // Rooms
    void roomDeleted(int32_t roomId);
    void joinedRoom(const QList<User>& allUsers, const QList<User>& activeUsers);
    void joinRoomFailed(const QString& error);
    void newRoomCreated(const RoomData& room, int32_t ownerId);
    void roomRenamed(int32_t roomId, const QString& newName);
    void becameMember();

    // Messages
    void newMessage(const MessageData& msg);
    void messagesLoaded(const QList<MessageData>& messages);

    // Users
    void userJoined(int32_t userId, const QString& username);
    void userLeft(int32_t userId, const QString& username);
    void userStartedTyping(int32_t userId, const QString& username);
    void userStoppedTyping(int32_t userId, const QString& username);
    void userRoleChanged(int32_t userId, chat::UserRights newRole);

    // Messages
    void messageDeleted(int32_t messageId);

    // Logout
    void loggedOut();

    // Account settings
    void changeUsernameSuccess();
    void changeUsernameFailure(const QString& error);
    void getMySaltResponse(bool success, const QString& salt);
    void changePasswordSuccess();
    void changePasswordFailure(const QString& error);
    void usernameChanged(int32_t userId, const QString& newUsername);

    // Generic
    void genericError(const QString& message);

private slots:
    void onConnected();
    void onDisconnected();
    void onBinaryMessage(const QByteArray& data);
    void onError(QAbstractSocket::SocketError error);

private:
    void sendEnvelope(const chat::Envelope& env);
    void handleMessage(const QByteArray& data);

    // --- Message handlers (called from handleMessage) ---
    void handleServerHello(const chat::Envelope& env);
    void handleInitialAuthResponse(const chat::Envelope& env);
    void handleAuthResponse(const chat::Envelope& env);
    void handleInitialRegisterResponse(const chat::Envelope& env);
    void handleRegisterResponse(const chat::Envelope& env);
    void handleGetServersResponse(const chat::Envelope& env);
    void handleServerAdded(const chat::Envelope& env);
    void handleServerRemoved(const chat::Envelope& env);
    void handleCreateRoomResponse(const chat::Envelope& env);
    void handleJoinRoomResponse(const chat::Envelope& env);
    void handleRoomMessage(const chat::Envelope& env);
    void handleGetMessagesResponse(const chat::Envelope& env);
    void handleNewRoomCreated(const chat::Envelope& env);
    void handleRoomDeleted(const chat::Envelope& env);
    void handleNewRoomName(const chat::Envelope& env);
    void handleUserJoined(const chat::Envelope& env);
    void handleUserLeft(const chat::Envelope& env);
    void handleLogoutResponse(const chat::Envelope& env);
    void handleSendMessageResponse(const chat::Envelope& env);
    void handleBecomeMemberResponse(const chat::Envelope& env);
    void handleUserStartedTyping(const chat::Envelope& env);
    void handleUserStoppedTyping(const chat::Envelope& env);
    void handleMessageDeleted(const chat::Envelope& env);
    void handleChangeUsernameResponse(const chat::Envelope& env);
    void handleGetMySaltResponse(const chat::Envelope& env);
    void handleChangePasswordResponse(const chat::Envelope& env);
    void handleUsernameChanged(const chat::Envelope& env);
    void handleUserTypingStartResponse(const chat::Envelope& env);
    void handleUserTypingStopResponse(const chat::Envelope& env);
    void handleAssignRoleResponse(const chat::Envelope& env);
    void handleUserRoleChanged(const chat::Envelope& env);
    void handleGenericError(const chat::Envelope& env);

    QWebSocket m_socket;
};

} // namespace qt_client
