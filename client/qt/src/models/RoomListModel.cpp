#include <models/RoomListModel.h>
#include <QDebug>

namespace qt_client {

RoomListModel::RoomListModel(QObject* parent)
    : QAbstractListModel(parent)
{}

int RoomListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid())
        return 0;
    return m_rooms.size();
}

QVariant RoomListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() >= m_rooms.size())
        return {};

    const auto& room = m_rooms.at(index.row());
    switch (role) {
    case RoomIdRole:       return room.id;
    case RoomNameRole:     return room.name;
    case IsJoinedRole:     return room.is_joined;
    case Qt::DisplayRole:  return room.name;
    default: return {};
    }
}

QHash<int, QByteArray> RoomListModel::roleNames() const {
    return {
        { RoomIdRole,      "roomId" },
        { RoomNameRole,    "roomName" },
        { IsJoinedRole,    "isJoined" },
    };
}

void RoomListModel::setRooms(QList<RoomData> rooms) {
    beginResetModel();
    m_rooms = std::move(rooms);
    endResetModel();
}

void RoomListModel::addRoom(const RoomData& room) {
    beginInsertRows(QModelIndex(), m_rooms.size(), m_rooms.size());
    m_rooms.append(room);
    endInsertRows();
}

void RoomListModel::removeRoom(int roomId) {
    for (int i = 0; i < m_rooms.size(); ++i) {
        if (m_rooms[i].id == roomId) {
            beginRemoveRows(QModelIndex(), i, i);
            m_rooms.removeAt(i);
            endRemoveRows();
            return;
        }
    }
}

void RoomListModel::clear() {
    beginResetModel();
    m_rooms.clear();
    endResetModel();
}

bool RoomListModel::isJoined(int32_t roomId) const {
    for (const auto& room : m_rooms) {
        if (room.id == roomId)
            return room.is_joined;
    }
    return false;
}

void RoomListModel::setJoined(int roomId) {
    for (int i = 0; i < m_rooms.size(); ++i) {
        if (m_rooms[i].id == roomId) {
            m_rooms[i].is_joined = true;
            emit dataChanged(index(i), index(i), { IsJoinedRole });
            return;
        }
    }
}

void RoomListModel::renameRoom(int roomId, const QString& newName) {
    for (int i = 0; i < m_rooms.size(); ++i) {
        if (m_rooms[i].id == roomId) {
            m_rooms[i].name = newName;
            emit dataChanged(index(i), index(i), { RoomNameRole });
            return;
        }
    }
}

} // namespace qt_client
