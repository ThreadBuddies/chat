#include <controllers/AppController.h>
#include <services/WsService.h>
#include <services/ConfigService.h>
#include <models/ServerListModel.h>
#include <models/RoomListModel.h>
#include <models/MessageListModel.h>
#include <common/utils/passwordUtil.h>
#include <utils/TextUtil.h>
#include <QtConcurrent>

namespace qt_client {

AppController::AppController(ConfigService* configService,
                             WebSocketService* wsService,
                             QObject* parent)
    : QObject(parent)
    , m_ws(wsService)
    , m_config(configService)
    , m_serverListModel(new ServerListModel(this))
    , m_roomListModel(new RoomListModel(this))
    , m_messageListModel(new MessageListModel(this))
    , m_userListModel(new UserListModel(this))
{
    m_localTypingTimer.setSingleShot(true);
    connect(&m_localTypingTimer, &QTimer::timeout, this, [this]() {
        m_isTyping = false;
        m_ws->sendTypingStop();
    });

    connectSignals();
}

// --- Property getters ---

AppController::Page AppController::currentPage() const { return m_currentPage; }
QString AppController::currentUsername() const { return m_currentUser.username; }
int32_t AppController::currentUserId() const{ return m_currentUser.id; }
QString AppController::currentRoomName() const { return m_currentRoom.name; }
int32_t AppController::currentRoomId() const { return m_currentRoom.id; }
QString AppController::errorMessage() const { return m_errorMessage; }

int AppController::currentUserRoomRole() const { return static_cast<int>(m_currentUser.role); }

int AppController::onlineCount() const {
    return m_userListModel->onlineCount();
}

QString AppController::typingUsers() const {
    if (m_typingUsers.isEmpty())
        return {};

    QStringList names;
    for (const auto& name : m_typingUsers)
        names.append(name);

    if (names.size() == 1)
        return names.first() + " is typing...";
    return names.join(", ") + " are typing...";
}

ServerListModel* AppController::serverListModel() const { return m_serverListModel; }
RoomListModel* AppController::roomListModel() const { return m_roomListModel; }
MessageListModel* AppController::messageListModel() const { return m_messageListModel; }
UserListModel* AppController::userListModel() const { return m_userListModel; }
ConfigService* AppController::configService() const { return m_config; }

void AppController::setCurrentPage(Page page) {
    if (m_currentPage != page) {
        m_currentPage = page;
        emit currentPageChanged();
    }
}

void AppController::setErrorMessage(const QString& msg) {
    if (m_errorMessage != msg) {
        m_errorMessage = msg;
        emit errorMessageChanged();
    }
}

void AppController::clearError() {
    setErrorMessage({});
}

void AppController::clearTypingState() {
    // Remote typing state
    for (auto* timer : m_typingTimers)
        timer->deleteLater();
    m_typingTimers.clear();

    bool hadTypers = !m_typingUsers.isEmpty();
    m_typingUsers.clear();
    if (hadTypers)
        emit typingUsersChanged();

    // Local typing state
    m_localTypingTimer.stop();
    m_isTyping = false;
}

// ============ Signal wiring ============

void AppController::connectSignals() {
    // Server discovery
    connect(m_ws, &WebSocketService::serverHelloReceived,
            this, &AppController::onServerHello);
    connect(m_ws, &WebSocketService::serversReceived,
            this, &AppController::onServersReceived);
    connect(m_ws, &WebSocketService::serverAdded,
            this, &AppController::onServerAdded);
    connect(m_ws, &WebSocketService::serverRemoved,
            this, &AppController::onServerRemoved);

    // Auth
    connect(m_ws, &WebSocketService::initialAuthResponse,
            this, &AppController::onInitialAuthResponse);
    connect(m_ws, &WebSocketService::authSuccess,
            this, &AppController::onAuthSuccess);
    connect(m_ws, &WebSocketService::authFailure,
            this, &AppController::onAuthFailure);

    // Registration
    connect(m_ws, &WebSocketService::initialRegisterResponse,
            this, &AppController::onInitialRegisterResponse);
    connect(m_ws, &WebSocketService::registerSuccess,
            this, &AppController::onRegisterSuccess);
    connect(m_ws, &WebSocketService::registerFailure,
            this, &AppController::onRegisterFailure);

    // Rooms
    connect(m_ws, &WebSocketService::roomDeleted,
            this, &AppController::onRoomDeleted);
    connect(m_ws, &WebSocketService::joinedRoom,
            this, &AppController::onJoinedRoom);
    connect(m_ws, &WebSocketService::joinRoomFailed,
            this, &AppController::onJoinRoomFailed);
    connect(m_ws, &WebSocketService::roomRenamed,
        this, &AppController::onRoomRenamed);

    connect(m_ws, &WebSocketService::becameMember,
            this, &AppController::onBecameMember);

    // Room broadcasts
    connect(m_ws, &WebSocketService::newRoomCreated,
            this, &AppController::onNewRoomCreated);

    // Users
    connect(m_ws, &WebSocketService::userJoined,
            this, &AppController::onUserJoined);
    connect(m_ws, &WebSocketService::userLeft,
            this, &AppController::onUserLeft);

    // Messages
    connect(m_ws, &WebSocketService::newMessage,
            this, &AppController::onNewMessage);
    connect(m_ws, &WebSocketService::messagesLoaded,
            this, &AppController::onMessagesLoaded);
    connect(m_ws, &WebSocketService::messageDeleted,
            this, &AppController::onMessageDeleted);

    // Typing
    connect(m_ws, &WebSocketService::userStartedTyping,
            this, &AppController::onUserStartedTyping);
    connect(m_ws, &WebSocketService::userStoppedTyping,
            this, &AppController::onUserStoppedTyping);

    // Role management
    connect(m_ws, &WebSocketService::userRoleChanged,
            this, &AppController::onUserRoleChanged);

    // Account settings
    connect(m_ws, &WebSocketService::changeUsernameSuccess,
            this, &AppController::onChangeUsernameSuccess);
    connect(m_ws, &WebSocketService::changeUsernameFailure,
            this, &AppController::onChangeUsernameFailure);
    connect(m_ws, &WebSocketService::getMySaltResponse,
            this, &AppController::onGetMySaltResponse);
    connect(m_ws, &WebSocketService::changePasswordSuccess,
            this, &AppController::onChangePasswordSuccess);
    connect(m_ws, &WebSocketService::changePasswordFailure,
            this, &AppController::onChangePasswordFailure);
    connect(m_ws, &WebSocketService::usernameChanged,
            this, &AppController::onUsernameChanged);

    // Lifecycle
    connect(m_ws, &WebSocketService::loggedOut,
            this, &AppController::onLoggedOut);
    connect(m_ws, &WebSocketService::error,
            this, &AppController::onWsError);
    connect(m_ws, &WebSocketService::genericError,
            this, &AppController::onGenericError);
    connect(m_ws, &WebSocketService::disconnected,
            this, &AppController::onDisconnected);
}

// ============ Server discovery handlers ============

void AppController::onServerHello(bool isAggregator) {
    if (isAggregator) {
        m_ws->sendGetServers();
    } else {
        setCurrentPage(Page::Auth);
    }
}

void AppController::onServersReceived(const QStringList& servers) {
    m_serverListModel->setServers(servers);
    setCurrentPage(Page::Servers);
}

void AppController::onServerAdded(const QString& host) {
    m_serverListModel->addServer(host);
}

void AppController::onServerRemoved(const QString& host) {
    m_serverListModel->removeServer(host);
}

// ============ Auth handlers ============

void AppController::onInitialAuthResponse(bool success, const QString& salt) {
    if (!success) {
        setErrorMessage("Failed to login");
        return;
    }

    std::string password = m_pendingAuth.password.toStdString();
    std::string serverSalt = salt.toStdString();
    m_pendingAuth.clear();

    struct HashResult {
        std::string hash;
        std::string salt;
        std::string password;
        bool isNewSalt;
    };

    QtConcurrent::run([password, serverSalt]() -> HashResult {
        HashResult res;
        res.isNewSalt = serverSalt.empty();
        res.salt = res.isNewSalt ? common::password::generate_salt() : serverSalt;
        res.hash = common::password::hash_password(password, res.salt);
        res.password = password;
        return res;

    }).then(this, [this](HashResult res) {
        if (res.isNewSalt) {
            m_ws->sendAuth(std::move(res.hash), std::move(res.password), std::move(res.salt));
        } else {
            m_ws->sendAuth(std::move(res.hash));
        }

    }).onFailed(this, [this](const std::exception& ex) {
        setErrorMessage(ex.what());
    });
}

void AppController::onAuthSuccess(User user, QList<RoomData> rooms) {
    m_currentUser = std::move(user);
    emit currentUsernameChanged();
    emit currentUserIdChanged();
    m_roomListModel->setRooms(std::move(rooms));
    setCurrentPage(Page::Chat);
}

void AppController::onAuthFailure(const QString& err) {
    setErrorMessage("Login failed: " + err);
}

void AppController::onInitialRegisterResponse(bool success, const QString& err) {
    if (!success) {
        setErrorMessage("Registration failed! " + err);
        return;
    }

    std::string password = m_pendingAuth.password.toStdString();

    struct HashResult {
        std::string hash;
        std::string salt;
    };

    QtConcurrent::run([password]() -> HashResult {
        HashResult res;
        res.salt = common::password::generate_salt();
        res.hash = common::password::hash_password(password, res.salt);
        return res;

    }).then(this, [this](HashResult res) {
        m_ws->sendRegister(std::move(res.salt), std::move(res.hash));

    }).onFailed(this, [this](const std::exception& ex) {
        setErrorMessage(ex.what());
        m_pendingAuth.clear();
    });
}

void AppController::onRegisterSuccess() {
    login(m_pendingAuth.username, m_pendingAuth.password);
}

void AppController::onRegisterFailure(const QString& err) {
    setErrorMessage("Registration failed! " + err);
    m_pendingAuth.clear();
}

// ============ Room handlers ============

void AppController::onRoomDeleted(int32_t roomId) {
    m_roomListModel->removeRoom(roomId);
    if (m_currentRoom.id == roomId) {
        m_currentRoom.id = -1;
        m_currentRoom.name.clear();
        emit currentRoomIdChanged();
        emit currentRoomNameChanged();
        m_messageListModel->clear();
        clearTypingState();
    }
}

void AppController::onJoinedRoom(QList<User> allUsers, QList<User> activeUsers) {
    m_optimisticMemberRoomId = -1;

    for (const auto& u : allUsers) {
        if (u.id == m_currentUser.id) {
            m_currentUser.role = u.role;
            emit currentUserRoomRoleChanged();
            break;
        }
    }

    m_userListModel->setUsers(std::move(allUsers));
    for (const auto& user : activeUsers) {
        m_userListModel->addUser(user);
    }
    emit onlineCountChanged();
    m_ws->sendGetMessages(MessageListModel::CHUNK_SIZE, MessageListModel::LATEST_TS);
}

void AppController::onJoinRoomFailed(const QString& err) {
    m_optimisticMemberRoomId = -1;
    m_currentRoom.id = -1;
    m_currentRoom.name.clear();
    emit currentRoomIdChanged();
    emit currentRoomNameChanged();
    m_messageListModel->clear();
    m_userListModel->clear();
    setErrorMessage("Failed to join room: " + err);
}

void AppController::onBecameMember() {
    int32_t roomId = m_optimisticMemberRoomId;
    m_optimisticMemberRoomId = -1;
    if (roomId < 0)
        return;
    m_roomListModel->setJoined(roomId);
    joinRoom(roomId);
}

void AppController::onNewRoomCreated(const RoomData& room, int32_t ownerId) {
    m_roomListModel->addRoom(room);
    if (m_currentUser.id == ownerId) {
        becomeMember(room.id);
    }
}

void AppController::onRoomRenamed(int32_t roomId, const QString& newName) {
    m_roomListModel->renameRoom(roomId, newName);
    if (m_currentRoom.id == roomId) {
        m_currentRoom.name = newName;
        emit currentRoomNameChanged();
    }
}

void AppController::onUserJoined(int32_t userId, const QString& username) {
    m_userListModel->addUser(User{userId, username, chat::REGULAR});
    emit onlineCountChanged();
}

void AppController::onUserLeft(int32_t userId, const QString& username) {
    m_userListModel->removeUser(userId);
    emit onlineCountChanged();
    onUserStoppedTyping(userId, username);
}

// ============ Message handlers ============

void AppController::onNewMessage(const MessageData& msg) {
    m_messageListModel->appendMessage(msg);
}

void AppController::onMessagesLoaded(const QList<MessageData>& messages) {
    m_messageListModel->onMessagesReceived(messages);
}

void AppController::onMessageDeleted(int32_t messageId) {
    m_messageListModel->deleteMessage(messageId);
}

// ============ Typing handlers ============

void AppController::onUserStartedTyping(int32_t userId, const QString& username) {
    if (userId == m_currentUser.id)
        return;

    m_typingUsers[userId] = username;

    // Start/restart 5s expiry timer
    if (m_typingTimers.contains(userId)) {
        m_typingTimers[userId]->start(5000);
    } else {
        auto* timer = new QTimer(this);
        timer->setSingleShot(true);
        connect(timer, &QTimer::timeout, this, [this, userId]() {
            m_typingUsers.remove(userId);
            m_typingTimers[userId]->deleteLater();
            m_typingTimers.remove(userId);
            emit typingUsersChanged();
        });
        m_typingTimers[userId] = timer;
        timer->start(5000);
    }

    emit typingUsersChanged();
}

void AppController::onUserStoppedTyping(int32_t userId, const QString& username) {
    Q_UNUSED(username);

    if (m_typingUsers.remove(userId)) {
        if (m_typingTimers.contains(userId)) {
            m_typingTimers[userId]->deleteLater();
            m_typingTimers.remove(userId);
        }
        emit typingUsersChanged();
    }
}

// ============ Lifecycle handlers ============

void AppController::resetSessionState() {
    m_currentUser.username.clear();
    m_currentUser.id = -1;
    m_currentUser.role = chat::UserRights::REGULAR;
    emit currentUsernameChanged();
    emit currentUserIdChanged();
    emit currentUserRoomRoleChanged();
    m_currentRoom.id = -1;
    m_currentRoom.name.clear();
    emit currentRoomIdChanged();
    emit currentRoomNameChanged();
    m_pendingAuth.clear();
    m_pendingOldPassword.clear();
    m_pendingNewPassword.clear();
    m_optimisticMemberRoomId = -1;
    m_roomListModel->clear();
    m_messageListModel->clear();
    m_userListModel->clear();
    clearTypingState();
}

void AppController::onLoggedOut() {
    resetSessionState();
    setCurrentPage(Page::Auth);
}

void AppController::onWsError(const QString& msg) {
    setErrorMessage(msg);
}

void AppController::onGenericError(const QString& msg) {
    setErrorMessage(msg);
}

void AppController::onDisconnected() {
    if (m_currentPage != Page::ServerList) {
        setErrorMessage("Connection lost");
        resetSessionState();
        setCurrentPage(Page::ServerList);
    }
}

void AppController::connectToAggregator(const QString& url) {
    clearError();
    m_ws->connectToUrl(url);
}

void AppController::connectToServer(const QString& host) {
    clearError();
    m_ws->disconnect();
    m_ws->connectToUrl(host);
}

void AppController::login(const QString& username, const QString& password) {
    clearError();

    QString sanitized = TextUtil::sanitizeInput(username);
    if (sanitized.isNull()) {
        setErrorMessage("Username must consist of at least 1 non-special character");
        return;
    }

    m_pendingAuth.username = sanitized;
    m_pendingAuth.password = password;
    m_pendingAuth.isRegistering = false;
    m_ws->sendInitialAuth(sanitized);
}

void AppController::registerUser(const QString& username, const QString& password) {
    clearError();

    QString sanitized = TextUtil::sanitizeInput(username);
    if (sanitized.isNull()) {
        setErrorMessage("Username must consist of at least 1 non-special character");
        return;
    }

    if (password.isEmpty()) {
        setErrorMessage("Password can't be empty");
        return;
    }

    m_pendingAuth.username = sanitized;
    m_pendingAuth.password = password;
    m_pendingAuth.isRegistering = true;
    m_ws->sendInitialRegister(sanitized);
}

void AppController::joinRoom(int32_t roomId) {
    clearError();
    clearTypingState();
    m_messageListModel->clear();

    // Find room name from model
    for (int32_t i = 0; i < m_roomListModel->rowCount(); ++i) {
        QModelIndex idx = m_roomListModel->index(i);
        if (m_roomListModel->data(idx, RoomListModel::RoomIdRole).toInt() == roomId) {
            m_currentRoom.name = m_roomListModel->data(idx, RoomListModel::RoomNameRole).toString();
            break;
        }
    }
    m_currentRoom.id = roomId;
    emit currentRoomIdChanged();
    emit currentRoomNameChanged();

    m_ws->sendJoinRoom(roomId);
}

void AppController::createRoom(const QString& name) {
    clearError();
    QString roomName = TextUtil::sanitizeInput(name);
    if (roomName.isNull()){
        setErrorMessage("Room name must consist of at least 1 non-special character");
        return;
    }
    m_ws->sendCreateRoom(roomName);
}

void AppController::sendMessage(const QString& text) {
    if (text.trimmed().isEmpty())
        return;
    m_ws->sendMessage(text);
}

void AppController::loadOlderMessages() {
    if (!m_messageListModel->canLoadMore() || m_messageListModel->isLoading())
        return;
    qint64 oldest = m_messageListModel->oldestTimestamp();
    if (oldest <= 0)
        return;
    m_messageListModel->beginLoadOlder();
    m_ws->sendGetMessages(MessageListModel::CHUNK_SIZE, oldest);
}

void AppController::loadNewerMessages() {
    if (!m_messageListModel->canLoadNewer() || m_messageListModel->isLoading())
        return;
    qint64 newest = m_messageListModel->newestTimestamp();
    if (newest <= 0)
        return;
    m_messageListModel->beginLoadNewer();
    m_ws->sendGetMessages(-MessageListModel::CHUNK_SIZE, newest);
}

void AppController::jumpToLatest() {
    m_messageListModel->clear();
    m_ws->sendGetMessages(MessageListModel::CHUNK_SIZE, MessageListModel::LATEST_TS);
}

void AppController::logout() {
    m_ws->sendLogout();
}

void AppController::disconnectFromServer() {
    m_ws->disconnect();
    resetSessionState();
    m_serverListModel->clear();
    setCurrentPage(Page::ServerList);
}

void AppController::goBack() {
    switch (m_currentPage) {
    case Page::Servers:
        m_ws->disconnect();
        setCurrentPage(Page::ServerList);
        break;
    case Page::Auth:
        m_ws->disconnect();
        setCurrentPage(Page::Servers);
        break;
    case Page::Chat:
        logout();
        break;
    case Page::AccountSettings:
        setCurrentPage(Page::Chat);
        break;
    default:
        break;
    }
}

void AppController::becomeMember(int32_t roomId) {
    clearError();
    m_optimisticMemberRoomId = roomId;
    m_ws->sendBecomeMember(roomId);
}

void AppController::startTyping() {
    if (!m_isTyping) {
        m_isTyping = true;
        m_ws->sendTypingStart();
    }
    m_localTypingTimer.start(2000);
}

void AppController::stopTyping() {
    m_localTypingTimer.stop();
    if (m_isTyping) {
        m_isTyping = false;
        m_ws->sendTypingStop();
    }
}

// ============ Account settings ============

void AppController::openAccountSettings() {
    setCurrentPage(Page::AccountSettings);
}

void AppController::changeUsername(const QString& newUsername) {
    QString sanitized = TextUtil::sanitizeInput(newUsername);
    if (sanitized.isNull()) {
        emit changeUsernameFailed("Username must consist of at least 1 non-special character");
        return;
    }
    if (sanitized == m_currentUser.username) {
        emit changeUsernameFailed("The new username is the same as the current one");
        return;
    }
    m_ws->sendChangeUsername(sanitized);
}

void AppController::changePassword(const QString& oldPassword, const QString& newPassword) {
    m_pendingOldPassword = oldPassword;
    m_pendingNewPassword = newPassword;
    m_ws->sendGetMySalt();
}

void AppController::onChangeUsernameSuccess() {
    emit changeUsernameSucceeded();
}

void AppController::onChangeUsernameFailure(const QString& error) {
    emit changeUsernameFailed("Failed to change username: " + error);
}

void AppController::onGetMySaltResponse(bool success, const QString& salt) {
    if (!success) {
        emit changePasswordFailed("Failed to retrieve data for password change");
        m_pendingOldPassword.clear();
        m_pendingNewPassword.clear();
        return;
    }

    std::string oldPassword = m_pendingOldPassword.toStdString();
    std::string newPassword = m_pendingNewPassword.toStdString();
    std::string currentSalt = salt.toStdString();
    m_pendingOldPassword.clear();
    m_pendingNewPassword.clear();

    struct HashResult {
        std::string oldHash;
        std::string newHash;
        std::string newSalt;
    };

    QtConcurrent::run([oldPassword, newPassword, currentSalt]() -> HashResult {
        HashResult res;
        res.oldHash = common::password::hash_password(oldPassword, currentSalt);
        res.newSalt = common::password::generate_salt();
        res.newHash = common::password::hash_password(newPassword, res.newSalt);
        return res;

    }).then(this, [this](HashResult res) {
        m_ws->sendChangePassword(std::move(res.oldHash), std::move(res.newHash), std::move(res.newSalt));

    }).onFailed(this, [this](const std::exception& ex) {
        emit changePasswordFailed(QString("Password change failed: ") + ex.what());
    });
}

void AppController::onChangePasswordSuccess() {
    emit changePasswordSucceeded();
}

void AppController::onChangePasswordFailure(const QString& error) {
    emit changePasswordFailed("Failed to change password: " + error);
}

void AppController::onUsernameChanged(int32_t userId, const QString& newUsername) {
    if (userId == m_currentUser.id) {
        m_currentUser.username = newUsername;
        emit currentUsernameChanged();
    }
    m_userListModel->updateUsername(userId, newUsername);
    m_messageListModel->updateUsername(userId, newUsername);
}

// ============ Role management ============

void AppController::assignRole(int userId, int newRole) {
    m_ws->sendAssignRole(m_currentRoom.id, userId, static_cast<chat::UserRights>(newRole));
}

void AppController::onUserRoleChanged(int32_t userId, chat::UserRights newRole) {
    m_userListModel->updateRole(userId, newRole);
    if (userId == m_currentUser.id) {
        m_currentUser.role = newRole;
        emit currentUserRoomRoleChanged();
    }
}

} // namespace qt_client
