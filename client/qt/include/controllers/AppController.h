#pragma once

#include <QObject>
#include <QString>
#include <QMap>
#include <QTimer>

#include <services/WsService.h>
#include <services/ConfigService.h>
#include <models/ServerListModel.h>
#include <models/RoomListModel.h>
#include <models/MessageListModel.h>
#include <models/UserListModel.h>
#include <models/User.h>

namespace qt_client {

class AppController : public QObject {
    Q_OBJECT

public:
    enum Page {
        ServerList = 0,
        Servers,
        Auth,
        Chat
    };
    Q_ENUM(Page)

    Q_PROPERTY(Page currentPage READ currentPage WRITE setCurrentPage NOTIFY currentPageChanged)
    Q_PROPERTY(QString currentUsername READ currentUsername NOTIFY currentUsernameChanged)
    Q_PROPERTY(int currentUserId READ currentUserId NOTIFY currentUserIdChanged)
    Q_PROPERTY(QString currentRoomName READ currentRoomName NOTIFY currentRoomNameChanged)
    Q_PROPERTY(int currentRoomId READ currentRoomId NOTIFY currentRoomIdChanged)
    Q_PROPERTY(QString errorMessage READ errorMessage NOTIFY errorMessageChanged)
    Q_PROPERTY(int onlineCount READ onlineCount NOTIFY onlineCountChanged)
    Q_PROPERTY(QString typingUsers READ typingUsers NOTIFY typingUsersChanged)

    Q_PROPERTY(ServerListModel* serverListModel READ serverListModel CONSTANT)
    Q_PROPERTY(RoomListModel* roomListModel READ roomListModel CONSTANT)
    Q_PROPERTY(MessageListModel* messageListModel READ messageListModel CONSTANT)
    Q_PROPERTY(UserListModel* userListModel READ userListModel CONSTANT)
    Q_PROPERTY(ConfigService* configService READ configService CONSTANT)

    explicit AppController(ConfigService* configService,
                           WebSocketService* wsService,
                           QObject* parent = nullptr);

    // Property getters
    Page currentPage() const;
    QString currentUsername() const;
    int32_t currentUserId() const;
    QString currentRoomName() const;
    int32_t currentRoomId() const;
    QString errorMessage() const;
    int onlineCount() const;
    QString typingUsers() const;

    ServerListModel* serverListModel() const;
    RoomListModel* roomListModel() const;
    MessageListModel* messageListModel() const;
    UserListModel* userListModel() const;
    ConfigService* configService() const;

    void setCurrentPage(Page page);

    // --- invokables ---
    Q_INVOKABLE void connectToAggregator(const QString& url);
    Q_INVOKABLE void connectToServer(const QString& host);
    Q_INVOKABLE void login(const QString& username, const QString& password);
    Q_INVOKABLE void registerUser(const QString& username, const QString& password);
    Q_INVOKABLE void joinRoom(int roomId);
    Q_INVOKABLE void createRoom(const QString& name);
    Q_INVOKABLE void sendMessage(const QString& text);
    Q_INVOKABLE void loadOlderMessages();
    Q_INVOKABLE void loadNewerMessages();
    Q_INVOKABLE void jumpToLatest();
    Q_INVOKABLE void logout();
    Q_INVOKABLE void disconnectFromServer();
    Q_INVOKABLE void goBack();
    Q_INVOKABLE void clearError();
    Q_INVOKABLE void becomeMember(int roomId);
    Q_INVOKABLE void startTyping();
    Q_INVOKABLE void stopTyping();

signals:
    void currentPageChanged();
    void currentUsernameChanged();
    void currentUserIdChanged();
    void currentRoomNameChanged();
    void currentRoomIdChanged();
    void errorMessageChanged();
    void onlineCountChanged();
    void typingUsersChanged();

private:
    void setErrorMessage(const QString& msg);
    void connectSignals();
    void resetSessionState();
    void clearTypingState();

    // --- Signal handlers ---
    void onServerHello(bool isAggregator);
    void onServersReceived(const QStringList& servers);
    void onServerAdded(const QString& host);
    void onServerRemoved(const QString& host);

    void onInitialAuthResponse(bool success, const QString& salt);
    void onAuthSuccess(User user, QList<RoomData> rooms);
    void onAuthFailure(const QString& err);

    void onInitialRegisterResponse(bool success, const QString& err);
    void onRegisterSuccess();
    void onRegisterFailure(const QString& err);

    void onRoomDeleted(int roomId);
    void onJoinedRoom(QList<User> allUsers, QList<User> activeUsers);
    void onNewRoomCreated(const RoomData& room, int32_t ownerId);
    void onRoomRenamed(int roomId, const QString& newName);
    void onJoinRoomFailed(const QString& error);
    void onBecameMember();
    void onUserJoined(int userId, const QString& username);
    void onUserLeft(int userId, const QString& username);

    void onNewMessage(const MessageData& msg);
    void onMessagesLoaded(const QList<MessageData>& messages);

    void onUserStartedTyping(int32_t userId, const QString& username);
    void onUserStoppedTyping(int32_t userId, const QString& username);
    void onMessageDeleted(int32_t messageId);

    void onLoggedOut();
    void onWsError(const QString& msg);
    void onGenericError(const QString& msg);
    void onDisconnected();

    WebSocketService* m_ws = nullptr;
    ConfigService* m_config = nullptr;

    ServerListModel* m_serverListModel = nullptr;
    RoomListModel* m_roomListModel = nullptr;
    MessageListModel* m_messageListModel = nullptr;
    UserListModel* m_userListModel = nullptr;

    Page m_currentPage = Page::ServerList;
    QString m_errorMessage;

    User m_currentUser;
    RoomData m_currentRoom;

    // Auth flow state
    struct PendingAuth {
        QString username;
        QString password;
        QString salt;
        bool isRegistering = false;
        void clear() { username.clear(); password.clear(); salt.clear(); isRegistering = false; }
    };
    PendingAuth m_pendingAuth;
    int32_t m_optimisticMemberRoomId = -1;

    // Typing indicator state (remote users)
    QMap<int32_t, QString> m_typingUsers;    // userId -> username
    QMap<int32_t, QTimer*> m_typingTimers;   // userId -> 5s expiry timer

    // Local typing state
    bool m_isTyping = false;
    QTimer m_localTypingTimer;
};

} // namespace qt_client
