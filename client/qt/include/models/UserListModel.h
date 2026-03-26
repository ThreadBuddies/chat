#pragma once

#include <QAbstractListModel>
#include <QList>
#include <models/User.h>

namespace qt_client {

class UserListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        UserIdRole = Qt::UserRole + 1,
        UserNameRole,
        UserRightsRole,
        CountRole
    };

    explicit UserListModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void setUsers(QList<User> users);
    void addUser(const User& user);
    void removeUser(int userId);
    void updateRole(int userId, chat::UserRights newRole);
    void updateUsername(int userId, const QString& newName);
    int onlineCount() const;
    void clear();

private:
    QList<User> m_users;
};

} // namespace qt_client
