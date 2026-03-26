#pragma once

#include <QAbstractListModel>
#include <QStringList>

namespace qt_client {

class ServerListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        HostRole = Qt::UserRole + 1
    };

    explicit ServerListModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void setServers(const QStringList& servers);
    void addServer(const QString& host);
    void removeServer(const QString& host);
    void clear();

private:
    QStringList m_servers;
};

} // namespace qt_client
