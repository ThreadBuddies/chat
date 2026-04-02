#include <services/WsService.h>
#include <common/version.h>
#include <common/proto/chat.pb.h>

namespace {

bool statusOk(const chat::Status& s) {
    return s.code() == chat::STATUS_SUCCESS;
}

QString statusMsg(const chat::Status& s) {
    return QString::fromStdString(s.message());
}

} // anonymous namespace

namespace qt_client {

WebSocketService::WebSocketService(QObject* parent)
    : QObject(parent)
{
    connect(&m_socket, &QWebSocket::connected,
            this, &WebSocketService::onConnected);
    connect(&m_socket, &QWebSocket::disconnected,
            this, &WebSocketService::onDisconnected);
    connect(&m_socket, &QWebSocket::binaryMessageReceived,
            this, &WebSocketService::onBinaryMessage);
    connect(&m_socket, &QWebSocket::errorOccurred,
            this, &WebSocketService::onError);
}

WebSocketService::~WebSocketService() {
    m_socket.close();
}

void WebSocketService::connectToUrl(const QString& url) {
    m_socket.close();
    qInfo() << "WebSocketService: connecting to" << url;

    QUrl qurl = QUrl::fromUserInput(url);
    if (!qurl.isValid()) {
        qInfo() << "WebSocketService: invalid URL:" << url;
        emit error(QStringLiteral("Invalid URL"));
        return;
    }

    m_socket.open(qurl);
}

void WebSocketService::disconnect() {
    m_socket.close();
}

// --- Connection slots ---

void WebSocketService::onConnected() {
    qInfo() << "WebSocketService: connected";
    emit connected();
}

void WebSocketService::onDisconnected() {
    qInfo() << "WebSocketService: disconnected";
    emit disconnected();
}

void WebSocketService::onBinaryMessage(const QByteArray& data) {
    handleMessage(data);
}

void WebSocketService::onError(QAbstractSocket::SocketError err) {
    qInfo() << "WebSocketService: socket error:" << err << m_socket.errorString();
    emit error(m_socket.errorString());
}

void WebSocketService::sendEnvelope(const chat::Envelope& env) {
    if (m_socket.state() != QAbstractSocket::SocketState::ConnectedState) {
        qInfo() << "WebSocketService: cannot send message, socket not connected";
        return;
    }

    QByteArray data;
    data.resize(env.ByteSizeLong());
    if (!env.SerializeToArray(data.data(), data.size())) {
        qInfo() << "WebSocketService: failed to serialize envelope";
        return;
    }

    qint64 bytesQueued = m_socket.sendBinaryMessage(data);
    if (bytesQueued != data.size()) {
        qInfo() << "WebSocketService: message not fully queued"
                    << bytesQueued << "/" << data.size();
    }
}

// ============ Message dispatch ============

void WebSocketService::handleMessage(const QByteArray& data) {
    chat::Envelope env;
    if (!env.ParseFromArray(data.constData(), data.size())) {
        qInfo() << "WebSocketService: failed to deserialize envelope";
        return;
    }

    using PC = chat::Envelope::PayloadCase;

    switch (env.payload_case()) {
        case PC::kServerHello:             handleServerHello(env); break;
        case PC::kInitialAuthResponse:     handleInitialAuthResponse(env); break;
        case PC::kAuthResponse:            handleAuthResponse(env); break;
        case PC::kInitialRegisterResponse: handleInitialRegisterResponse(env); break;
        case PC::kRegisterResponse:        handleRegisterResponse(env); break;
        case PC::kGetServersResponse:      handleGetServersResponse(env); break;
        case PC::kServerAdded:             handleServerAdded(env); break;
        case PC::kServerRemoved:           handleServerRemoved(env); break;
        case PC::kCreateRoomResponse:      handleCreateRoomResponse(env); break;
        case PC::kJoinRoomResponse:        handleJoinRoomResponse(env); break;
        case PC::kRoomMessage:             handleRoomMessage(env); break;
        case PC::kGetMessagesResponse:     handleGetMessagesResponse(env); break;
        case PC::kNewRoomCreated:          handleNewRoomCreated(env); break;
        case PC::kRoomDeleted:             handleRoomDeleted(env); break;
        case PC::kUserJoined:              handleUserJoined(env); break;
        case PC::kUserLeft:                handleUserLeft(env); break;
        case PC::kLogoutResponse:          handleLogoutResponse(env); break;
        case PC::kSendMessageResponse:     handleSendMessageResponse(env); break;
        case PC::kGenericError:            handleGenericError(env); break;
        case PC::kUserStartedTyping:       handleUserStartedTyping(env); break;
        case PC::kUserStoppedTyping:       handleUserStoppedTyping(env); break;
        case PC::kNewRoomName:             handleNewRoomName(env); break;
        case PC::kMessageDeleted:          handleMessageDeleted(env); break;
        case PC::kBecomeMemberResponse:    handleBecomeMemberResponse(env); break;
        case PC::kChangeUsernameResponse:  handleChangeUsernameResponse(env); break;
        case PC::kGetMySaltResponse:       handleGetMySaltResponse(env); break;
        case PC::kChangePasswordResponse:  handleChangePasswordResponse(env); break;
        case PC::kUsernameChanged:         handleUsernameChanged(env); break;
        case PC::kAssignRoleResponse:      handleAssignRoleResponse(env); break;
        case PC::kUserRoleChanged:         handleUserRoleChanged(env); break;
        default: qDebug() << "WebSocketService: unhandled envelope payload"; break;
    }
}

// ============ Auth handlers ============

void WebSocketService::handleServerHello(const chat::Envelope& env) {
    const auto& hello = env.server_hello();
    if (hello.protocol_version() != common::version::PROTOCOL_VERSION) {
        qInfo() << "Protocol version mismatch: server=" << hello.protocol_version()
                   << "client=" << common::version::PROTOCOL_VERSION;
        emit error(QStringLiteral("Version mismatch, update your client"));
        m_socket.close();
        return;
    }
    bool isAggregator = (hello.type() == chat::ServerType::TYPE_AGGREGATOR);
    emit serverHelloReceived(isAggregator);
}

void WebSocketService::handleInitialAuthResponse(const chat::Envelope& env) {
    const auto& resp = env.initial_auth_response();
    bool success = statusOk(resp.status());
    QString salt = resp.has_salt() ? QString::fromStdString(resp.salt()) : QString();
    emit initialAuthResponse(success, salt);
}

void WebSocketService::handleAuthResponse(const chat::Envelope& env) {
    const auto& resp = env.auth_response();
    if (statusOk(resp.status())) {
        QList<RoomData> rooms;
        for (const auto& ri : resp.rooms()) {
            rooms.emplace_back(
                ri.room_id(),
                QString::fromStdString(ri.room_name()),
                ri.is_joined());
        }
        qt_client::User user(resp.authenticated_user().user_id(),
                             QString::fromStdString(resp.authenticated_user().user_name()),
                             chat::UserRights::REGULAR);

        emit authSuccess(std::move(user), std::move(rooms));
    } else {
        emit authFailure(statusMsg(resp.status()));
    }
}

void WebSocketService::handleInitialRegisterResponse(const chat::Envelope& env) {
    const auto& resp = env.initial_register_response();
    bool success = statusOk(resp.status());
    emit initialRegisterResponse(success, success ? QString() : statusMsg(resp.status()));
}

void WebSocketService::handleRegisterResponse(const chat::Envelope& env) {
    const auto& resp = env.register_response();
    if (statusOk(resp.status()))
        emit registerSuccess();
    else
        emit registerFailure(statusMsg(resp.status()));
}

// ============ Server discovery handlers ============

void WebSocketService::handleGetServersResponse(const chat::Envelope& env) {
    const auto& resp = env.get_servers_response();
    QStringList servers;
    for (const auto& s : resp.servers()) {
        servers.append(QString::fromStdString(s.host()));
    }
    emit serversReceived(servers);
}

void WebSocketService::handleServerAdded(const chat::Envelope& env) {
    emit serverAdded(QString::fromStdString(env.server_added().server().host()));
}

void WebSocketService::handleServerRemoved(const chat::Envelope& env) {
    emit serverRemoved(QString::fromStdString(env.server_removed().server().host()));
}

// ============ Room handlers ============

void WebSocketService::handleCreateRoomResponse(const chat::Envelope& env) {
    const auto& resp = env.create_room_response();
    if (!statusOk(resp.status()))
        emit error(statusMsg(resp.status()));
}

void WebSocketService::handleJoinRoomResponse(const chat::Envelope& env) {
    const auto& resp = env.join_room_response();
    if (statusOk(resp.status())) {
        QList<User> allUsers;
        allUsers.reserve(env.join_room_response().all_users().size());
        
        for (const auto& u : resp.all_users()) {
            allUsers.emplace_back(
                u.user_id(), 
                QString::fromStdString(u.user_name()),
                u.user_room_rights());
        }
        QList<User> activeUsers;
        activeUsers.reserve(resp.active_users().size());
        for (const auto& u : resp.active_users()) {
            activeUsers.emplace_back(
                u.user_id(),
                QString::fromStdString(u.user_name()),
                u.user_room_rights());
        }
        emit joinedRoom(std::move(allUsers), std::move(activeUsers));
    } else {
        emit joinRoomFailed(statusMsg(resp.status()));
    }
}

void WebSocketService::handleNewRoomCreated(const chat::Envelope& env) {
    const auto& ri = env.new_room_created().room();
    int32_t ownerId = ri.owner().user_id();
    RoomData room{ri.room_id(),
                  QString::fromStdString(ri.room_name()), 
                  ri.is_joined()};
    emit newRoomCreated(room, ownerId);
}

void WebSocketService::handleNewRoomName(const chat::Envelope& env) {
    const auto& msg = env.new_room_name();
    emit roomRenamed(msg.room_id(), QString::fromStdString(msg.name()));
}

void WebSocketService::handleBecomeMemberResponse(const chat::Envelope& env) {
    const auto& resp = env.become_member_response();
    if (statusOk(resp.status())) {
        emit becameMember();
    } else {
        emit error(statusMsg(resp.status()));
    }
}

void WebSocketService::handleRoomDeleted(const chat::Envelope& env) {
    emit roomDeleted(env.room_deleted().room_id());
}

// ============ Message handlers ============

void WebSocketService::handleRoomMessage(const chat::Envelope& env) {
    const auto& mi = env.room_message().message();
    MessageData msg;
    msg.user = QString::fromStdString(mi.from().user_name());
    msg.userId = mi.from().user_id();
    msg.msg = QString::fromStdString(mi.message());
    msg.timestamp = mi.timestamp();
    msg.messageId = mi.message_id();
    emit newMessage(msg);
}

void WebSocketService::handleGetMessagesResponse(const chat::Envelope& env) {
    const auto& resp = env.get_messages_response();
    QList<MessageData> messages;
    for (const auto& mi : resp.message()) {
        MessageData msg;
        msg.user = QString::fromStdString(mi.from().user_name());
        msg.userId = mi.from().user_id();
        msg.msg = QString::fromStdString(mi.message());
        msg.timestamp = mi.timestamp();
        msg.messageId = mi.message_id();
        messages.append(msg);
    }
    emit messagesLoaded(messages);
}

void WebSocketService::handleSendMessageResponse(const chat::Envelope& env) {
    const auto& resp = env.send_message_response();
    if (!statusOk(resp.status()))
        emit error(statusMsg(resp.status()));
}

// ============ User handlers ============

void WebSocketService::handleUserJoined(const chat::Envelope& env) {
    const auto& u = env.user_joined().user();
    emit userJoined(u.user_id(), QString::fromStdString(u.user_name()));
}

void WebSocketService::handleUserLeft(const chat::Envelope& env) {
    const auto& u = env.user_left().user();
    emit userLeft(u.user_id(), QString::fromStdString(u.user_name()));
}

void WebSocketService::handleLogoutResponse(const chat::Envelope& env) {
    const auto& resp = env.logout_response();
    if (statusOk(resp.status())) {
        emit loggedOut();
    } else {
        emit error(statusMsg(resp.status()));
    }
}

// ============ Typing handlers ============

void WebSocketService::handleUserStartedTyping(const chat::Envelope& env) {
    const auto& u = env.user_started_typing().user();
    emit userStartedTyping(u.user_id(), QString::fromStdString(u.user_name()));
}

void WebSocketService::handleUserStoppedTyping(const chat::Envelope& env) {
    const auto& u = env.user_stopped_typing().user();
    emit userStoppedTyping(u.user_id(), QString::fromStdString(u.user_name()));
}

// ============  message deletion handler ============

void WebSocketService::handleMessageDeleted(const chat::Envelope& env) {
    emit messageDeleted(env.message_deleted().message_id());
}

// ============ Account settings handlers ============

void WebSocketService::handleChangeUsernameResponse(const chat::Envelope& env) {
    const auto& resp = env.change_username_response();
    if (statusOk(resp.status()))
        emit changeUsernameSuccess();
    else
        emit changeUsernameFailure(statusMsg(resp.status()));
}

void WebSocketService::handleGetMySaltResponse(const chat::Envelope& env) {
    const auto& resp = env.get_my_salt_response();
    bool success = statusOk(resp.status());
    QString salt = resp.has_salt() ? QString::fromStdString(resp.salt()) : QString();
    emit getMySaltResponse(success, salt);
}

void WebSocketService::handleChangePasswordResponse(const chat::Envelope& env) {
    const auto& resp = env.change_password_response();
    if (statusOk(resp.status()))
        emit changePasswordSuccess();
    else
        emit changePasswordFailure(statusMsg(resp.status()));
}

void WebSocketService::handleUsernameChanged(const chat::Envelope& env) {
    const auto& msg = env.username_changed();
    emit usernameChanged(msg.user_id(), QString::fromStdString(msg.new_username()));
}

// ============ Role handlers ============

void WebSocketService::handleAssignRoleResponse(const chat::Envelope& env) {
    const auto& resp = env.assign_role_response();
    if (!statusOk(resp.status()))
        emit error(statusMsg(resp.status()));
}

void WebSocketService::handleUserRoleChanged(const chat::Envelope& env) {
    const auto& msg = env.user_role_changed();
    emit userRoleChanged(msg.user_id(), msg.new_role());
}

// ============ Error handlers ============

void WebSocketService::handleGenericError(const chat::Envelope& env) {
    emit genericError(statusMsg(env.generic_error().status()));
}

// --- Send methods ---

void WebSocketService::sendGetServers() {
    chat::Envelope env;
    env.mutable_get_servers_request();
    sendEnvelope(env);
}

void WebSocketService::sendInitialAuth(const QString& username) {
    chat::Envelope env;
    env.mutable_initial_auth_request()->set_username(username.toStdString());
    sendEnvelope(env);
}

void WebSocketService::sendAuth(std::string&& hash, std::optional<std::string>&& password, std::optional<std::string>&& salt) {
    chat::Envelope env;
    auto* req = env.mutable_auth_request();
    req->set_hash(std::move(hash));
    if (password) req->set_password(std::move(*password));
    if (salt) req->set_salt(std::move(*salt));
    sendEnvelope(env);
}

void WebSocketService::sendInitialRegister(const QString& username) {
    chat::Envelope env;
    env.mutable_initial_register_request()->set_username(username.toStdString());
    sendEnvelope(env);
}

void WebSocketService::sendRegister(std::string&& salt, std::string&& hash) {
    chat::Envelope env;
    auto* req = env.mutable_register_request();
    req->set_salt(std::move(salt));
    req->set_hash(std::move(hash));
    sendEnvelope(env);
}

void WebSocketService::sendJoinRoom(int roomId) {
    chat::Envelope env;
    env.mutable_join_room_request()->set_room_id(roomId);
    sendEnvelope(env);
}

void WebSocketService::sendCreateRoom(const QString& name) {
    chat::Envelope env;
    env.mutable_create_room_request()->set_room_name(name.toStdString());
    sendEnvelope(env);
}

void WebSocketService::sendMessage(const QString& text) {
    chat::Envelope env;
    env.mutable_send_message_request()->set_message(text.toStdString());
    sendEnvelope(env);
}

void WebSocketService::sendGetMessages(int limit, qint64 offsetTs) {
    chat::Envelope env;
    auto* req = env.mutable_get_messages_request();
    req->set_limit(limit);
    req->set_offset_ts(offsetTs);
    sendEnvelope(env);
}

void WebSocketService::sendLogout() {
    chat::Envelope env;
    env.mutable_logout_request();
    sendEnvelope(env);
}

void WebSocketService::sendTypingStart() {
    chat::Envelope env;
    env.mutable_user_typing_start_request();
    sendEnvelope(env);
}

void WebSocketService::sendTypingStop() {
    chat::Envelope env;
    env.mutable_user_typing_stop_request();
    sendEnvelope(env);
}

void WebSocketService::sendBecomeMember(int roomId) {
    chat::Envelope env;
    env.mutable_become_member_request()->set_room_id(roomId);
    sendEnvelope(env);
}

void WebSocketService::sendChangeUsername(const QString& newUsername) {
    chat::Envelope env;
    env.mutable_change_username_request()->set_new_username(newUsername.toStdString());
    sendEnvelope(env);
}

void WebSocketService::sendGetMySalt() {
    chat::Envelope env;
    env.mutable_get_my_salt_request();
    sendEnvelope(env);
}

void WebSocketService::sendChangePassword(std::string oldHash, std::string newHash, std::string newSalt) {
    chat::Envelope env;
    auto* req = env.mutable_change_password_request();
    req->set_old_password_hash(std::move(oldHash));
    req->set_new_password_hash(std::move(newHash));
    req->set_new_salt(std::move(newSalt));
    sendEnvelope(env);
}

void WebSocketService::sendAssignRole(int32_t roomId, int32_t userId, chat::UserRights role) {
    chat::Envelope env;
    auto* req = env.mutable_assign_role_request();
    req->set_room_id(roomId);
    req->set_user_id(userId);
    req->set_new_role(role);
    sendEnvelope(env);
}

} // namespace qt_client
