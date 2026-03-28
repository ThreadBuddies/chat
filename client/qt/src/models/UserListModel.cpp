#include <models/UserListModel.h>
#include <QSet>

namespace qt_client {

UserListModel::UserListModel(QObject* parent)
    : QAbstractListModel(parent)
{}

int UserListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    return m_users.size();
}

QVariant UserListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() >= m_users.size())
        return {};

    const auto& user = m_users[index.row()];
    switch (role) {
        case UserIdRole:     return user.id;
        case UserNameRole:   return user.username;
        case UserRightsRole: return static_cast<int>(user.role);
        case CountRole:      return user.count;
        default:             return {};
    }
}

QHash<int, QByteArray> UserListModel::roleNames() const {
    return {
        {UserIdRole,     "userId"},
        {UserNameRole,   "userName"},
        {UserRightsRole, "userRights"},
        {CountRole,      "count"}
    };
}

void UserListModel::setUsers(QList<User> users) {
    beginResetModel();
    // Sort the incoming user list by the User::operator<
    std::sort(users.begin(), users.end(), [] (const User& lhs, const User& rhs) {
        return !(lhs < rhs);
    });
    m_users = std::move(users);
    endResetModel();
}

void UserListModel::addUser(const User& user) {
    auto it = std::find_if(m_users.begin(), m_users.end(), [userId = user.id](const User& u) {
		return u.id == userId; });
    if (it != m_users.end()) {
        ++it->count;
    } else {
        m_users.emplace_back(user.id, user.username, user.role);
        m_users.back().count = 1;
    }
    setUsers(std::move(m_users));
}

void UserListModel::removeUser(int userId) {
    auto it = std::find_if(m_users.begin(), m_users.end(), [userId](const User& u) {
        return u.id == userId; });

    if (it != m_users.end()) {
        if (it->count > 0) {
            --it->count;
        } else {
            it = m_users.erase(it);
		}
        setUsers(std::move(m_users));
    }
}

void UserListModel::updateRole(int userId, chat::UserRights newRole) {
    std::for_each(m_users.begin(), m_users.end(), [userId, newRole](User& u) {
        if(u.id == userId) {
            u.role = newRole;
        }
    });

    setUsers(std::move(m_users));
}

void UserListModel::updateUsername(int userId, const QString& newName) {
    std::for_each(m_users.begin(), m_users.end(), [userId, newName](User& u) {
        if (u.id == userId) {
            u.username = newName;
        }
    });

    setUsers(std::move(m_users));
}

int UserListModel::onlineCount() const {
    int count = 0;
    for (const auto& u : m_users) {
        if (u.count > 0)
            ++count;
    }
    return count;
}

void UserListModel::clear() {
    beginResetModel();
    m_users.clear();
    endResetModel();
}

} // namespace qt_client
