#pragma once

#include <QAbstractListModel>
#include <QList>
#include <QString>
#include <limits>

namespace qt_client {

struct MessageData {
    QString user;
    qint32 userId;
    QString msg;
    qint64 timestamp;
    qint32 messageId;
};

class MessageListModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(bool canLoadMore READ canLoadMore NOTIFY canLoadMoreChanged)
    Q_PROPERTY(bool canLoadNewer READ canLoadNewer NOTIFY canLoadNewerChanged)

public:
    static constexpr int MAX_MESSAGES = 100;
    static constexpr int CHUNK_SIZE = 25;
    static constexpr qint64 LATEST_TS = std::numeric_limits<qint64>::max();

    enum Roles {
        UsernameRole = Qt::UserRole + 1,
        UserIdRole,
        MessageTextRole,
        TimestampRole,
        MessageIdRole,
        FormattedTimeRole
    };

    explicit MessageListModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void appendMessage(const MessageData& msg);
    void onMessagesReceived(const QList<MessageData>& messages);
    void clear();
    void deleteMessage(int messageId);
    void updateUsername(int userId, const QString& newName);

    bool canLoadMore() const;
    bool canLoadNewer() const;

    bool isLoading() const;
    void beginLoadOlder();
    void beginLoadNewer();

    qint64 oldestTimestamp() const;
    qint64 newestTimestamp() const;



signals:
    void canLoadMoreChanged();
    void canLoadNewerChanged();

private:
    void prependMessages(const QList<MessageData>& messages);
    void appendMessages(const QList<MessageData>& messages);
    void setCanLoadMore(bool value);
    void setCanLoadNewer(bool value);

    QList<MessageData> m_messages;
    bool m_canLoadMore = true;
    bool m_canLoadNewer = false;
    bool m_loadingOlder = false;
    bool m_loadingNewer = false;

    static QString formatTimestamp(qint64 timestamp);
};

} // namespace qt_client
