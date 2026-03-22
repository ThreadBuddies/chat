#include "models/message_list_model.h"
#include <QDateTime>
#include <QDebug>

namespace qt_client {

MessageListModel::MessageListModel(QObject* parent)
    : QAbstractListModel(parent)
{}

int MessageListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid())
        return 0;
    return m_messages.size();
}

QVariant MessageListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() >= m_messages.size())
        return {};

    const auto& msg = m_messages.at(index.row());
    switch (role) {
    case UsernameRole:      return msg.user;
    case UserIdRole:        return msg.userId;
    case MessageTextRole:   return msg.msg;
    case TimestampRole:     return msg.timestamp;
    case MessageIdRole:     return msg.messageId;
    case FormattedTimeRole: return formatTimestamp(msg.timestamp);
    case Qt::DisplayRole:   return msg.msg;
    default: return {};
    }
}

QHash<int, QByteArray> MessageListModel::roleNames() const {
    return {
        { UsernameRole,      "username" },
        { UserIdRole,        "userId" },
        { MessageTextRole,   "messageText" },
        { TimestampRole,     "timestamp" },
        { MessageIdRole,     "messageId" },
        { FormattedTimeRole, "formattedTime" }
    };
}

void MessageListModel::appendMessage(const MessageData& msg) {
    beginInsertRows(QModelIndex(), 0, 0);
    m_messages.prepend(msg);
    endInsertRows();

    // Trim oldest (top) if over limit
    if (m_messages.size() > MAX_MESSAGES) {
        int excess = m_messages.size() - MAX_MESSAGES;
        int last = m_messages.size() - 1;
        beginRemoveRows(QModelIndex(), last - excess + 1, last);
        m_messages.erase(m_messages.end() - excess, m_messages.end());
        endRemoveRows();
    }
}

void MessageListModel::onMessagesReceived(const QList<MessageData>& messages) {
    if (m_loadingNewer) {
        prependMessages(messages);
    } else {
        appendMessages(messages);
    }
    m_loadingOlder = false;
    m_loadingNewer = false;
}

bool MessageListModel::isLoading() const {
    return m_loadingOlder || m_loadingNewer;
}

void MessageListModel::beginLoadOlder() {
    m_loadingOlder = true;
}

void MessageListModel::beginLoadNewer() {
    m_loadingNewer = true;
}

// Insert older messages at the end (top of BottomToTop view)
// Trim newest from beginning (bottom) if over limit
void MessageListModel::appendMessages(const QList<MessageData>& messages) {
    if (messages.isEmpty())
        return;

    int first = m_messages.size();
    beginInsertRows(QModelIndex(), first, first + messages.size() - 1);
    for (const auto& msg : messages) {
        m_messages.append(msg);
    }
    endInsertRows();

    if (static_cast<int>(messages.size()) < CHUNK_SIZE) {
        setCanLoadMore(false);
    }

    // Trim newest (bottom) if over limit
    if (m_messages.size() > MAX_MESSAGES) {
        int excess = m_messages.size() - MAX_MESSAGES;
        beginRemoveRows(QModelIndex(), 0, excess - 1);
        m_messages.erase(m_messages.begin(), m_messages.begin() + excess);
        endRemoveRows();
        setCanLoadNewer(true);
    }
}

// Insert newer messages at the beginning (bottom of BottomToTop view)
// Trim oldest from end (top) if over limit
void MessageListModel::prependMessages(const QList<MessageData>& messages) {
    if (messages.isEmpty())
        return;

    beginInsertRows(QModelIndex(), 0, messages.size() - 1);
    for (int i = 0; i < messages.size(); ++i) {
        m_messages.prepend(messages[i]);
    }
    endInsertRows();

    if (static_cast<int>(messages.size()) < CHUNK_SIZE) {
        setCanLoadNewer(false);
    }

    // Trim oldest (top) if over limit
    if (m_messages.size() > MAX_MESSAGES) {
        int excess = m_messages.size() - MAX_MESSAGES;
        int last = m_messages.size() - 1;
        beginRemoveRows(QModelIndex(), last - excess + 1, last);
        m_messages.erase(m_messages.end() - excess, m_messages.end());
        endRemoveRows();
        setCanLoadMore(true);
    }
}

void MessageListModel::clear() {
    beginResetModel();
    m_messages.clear();
    m_canLoadMore = true;
    m_canLoadNewer = false;
    endResetModel();
    emit canLoadMoreChanged();
    emit canLoadNewerChanged();
}

bool MessageListModel::canLoadMore() const {
    return m_canLoadMore;
}

void MessageListModel::deleteMessage(int messageId) {
    for (int i = 0; i < m_messages.size(); ++i) {
        if (m_messages[i].messageId == messageId) {
            beginRemoveRows(QModelIndex(), i, i);
            m_messages.removeAt(i);
            endRemoveRows();
            return;
        }
    }
}

void MessageListModel::setCanLoadMore(bool value) {
    if (m_canLoadMore != value) {
        m_canLoadMore = value;
        emit canLoadMoreChanged();
    }
}

bool MessageListModel::canLoadNewer() const {
    return m_canLoadNewer;
}

void MessageListModel::setCanLoadNewer(bool value) {
    if (m_canLoadNewer != value) {
        m_canLoadNewer = value;
        emit canLoadNewerChanged();
    }
}

qint64 MessageListModel::oldestTimestamp() const {
    if (m_messages.isEmpty())
        return LATEST_TS;
    return m_messages.last().timestamp;
}

qint64 MessageListModel::newestTimestamp() const {
    if (m_messages.isEmpty())
        return 0;
    return m_messages.first().timestamp;
}

QString MessageListModel::formatTimestamp(qint64 timestamp) {
    QDateTime dt = QDateTime::fromMSecsSinceEpoch(timestamp / 1000);
    QDate today = QDate::currentDate();
    QDate msgDate = dt.date();

    if (msgDate == today) {
        return "[" + dt.toString("HH:mm") + "]";
    }
    if (msgDate.year() == today.year()) {
        return "[" + dt.toString("dd.MM HH:mm") + "]";
    }
    return "[" + dt.toString("dd.MM.yyyy HH:mm") + "]";
}

} // namespace qt_client
