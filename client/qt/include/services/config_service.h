#pragma once

#include <QObject>
#include <QStringList>
#include <QString>
#include <QJsonObject>

class QLockFile;

namespace qt_client {

class ConfigService : public QObject {
    Q_OBJECT
    Q_PROPERTY(QStringList servers READ servers NOTIFY serversChanged)

public:
    // The constructor takes app's name to create a unique directory
    // and instance lock name.
    explicit ConfigService(const QString& appName, QObject* parent = nullptr);

    // The destructor handles saving the config and releasing the instance lock.
    ~ConfigService() override;

    // Expose the root JSON object for reading and writing.
    QJsonObject& getRoot();
    const QJsonObject& getRoot() const;

    // Check if this is the first instance of the application.
    bool isFirstInstance() const;

    // Returns a list of unique server strings.
    QStringList servers() const;

    // Adds a server if it's not already present. Returns true if added.
    Q_INVOKABLE bool addServer(const QString& server);

    // Removes all occurrences of a server.
    Q_INVOKABLE void removeServer(const QString& server);

signals:
    void serversChanged();

private:
    // Helper to initialize the path. Called from the constructor.
    void initPath();
    // Helper to read the config file. Called from the constructor.
    void readConfig();

    QJsonObject m_root;
    QString m_configFilePath;
    QLockFile* m_lockFile;
    bool m_isFirstInstance;

    // In-memory cache for the server list. Synced on load and save.
    QStringList m_servers;

    // Disable copy constructor and assignment operator
    ConfigService(const ConfigService&) = delete;
    ConfigService& operator=(const ConfigService&) = delete;
};

} // namespace qt_client
