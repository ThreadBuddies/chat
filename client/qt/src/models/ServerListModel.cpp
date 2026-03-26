#include <models/ServerListModel.h>

namespace qt_client {

ServerListModel::ServerListModel(QObject* parent)
    : QAbstractListModel(parent)
{}

int ServerListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid())
        return 0;
    return m_servers.size();
}

QVariant ServerListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() >= m_servers.size())
        return {};

    if (role == HostRole || role == Qt::DisplayRole)
        return m_servers.at(index.row());

    return {};
}

QHash<int, QByteArray> ServerListModel::roleNames() const {
    return {
        { HostRole, "host" }
    };
}

void ServerListModel::setServers(const QStringList& servers) {
    beginResetModel();
    m_servers = servers;
    endResetModel();
}

void ServerListModel::addServer(const QString& host) {
    beginInsertRows(QModelIndex(), m_servers.size(), m_servers.size());
    m_servers.append(host);
    endInsertRows();
}

void ServerListModel::removeServer(const QString& host) {
    int idx = m_servers.indexOf(host);
    if (idx < 0)
        return;
    beginRemoveRows(QModelIndex(), idx, idx);
    m_servers.removeAt(idx);
    endRemoveRows();
}

void ServerListModel::clear() {
    beginResetModel();
    m_servers.clear();
    endResetModel();
}

} // namespace qt_client
