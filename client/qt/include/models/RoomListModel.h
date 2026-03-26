#pragma once

#include <QAbstractListModel>
#include <QList>
#include <QString>

namespace qt_client {

struct RoomData {
    int32_t id = -1;
    QString name;
    bool is_joined = false;
};

class RoomListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        RoomIdRole = Qt::UserRole + 1,
        RoomNameRole,
        IsJoinedRole,
    };

    explicit RoomListModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void setRooms(QList<RoomData> rooms);
    void addRoom(const RoomData& room);
    void removeRoom(int roomId);
    void clear();
    void renameRoom(int roomId, const QString& newName);
    void setJoined(int roomId);
    bool isJoined(int32_t roomId) const;

private:
    QList<RoomData> m_rooms;
};

} // namespace qt_client
