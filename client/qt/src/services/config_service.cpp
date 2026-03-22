#include "services/config_service.h"

#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLockFile>
#include <QDebug>

namespace qt_client {

ConfigService::ConfigService(const QString& appName, QObject* parent)
    : QObject(parent)
    , m_lockFile(nullptr)
    , m_isFirstInstance(false)
{
    QString lockPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation)
                       + "/" + appName + ".lock";
    m_lockFile = new QLockFile(lockPath);

    if (m_lockFile->tryLock()) {
        m_isFirstInstance = true;
    } else {
        m_isFirstInstance = false;
        qWarning() << "Another instance of the application is already running.";
    }

    initPath();
    readConfig();
}

ConfigService::~ConfigService() {
    if (m_isFirstInstance) {
        // Sync the in-memory server list back to the JSON object before saving.
        QJsonArray serversArray;
        for (const auto& server : m_servers) {
            serversArray.append(server);
        }
        m_root["servers"] = serversArray;

        QFile file(m_configFilePath);
        if (!file.open(QIODevice::WriteOnly)) {
            qCritical() << "Failed to open config file for writing:" << m_configFilePath;
        } else {
            file.write(QJsonDocument(m_root).toJson(QJsonDocument::Indented));
        }
    }
    delete m_lockFile;
}

void ConfigService::initPath() {
    QString configDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);

    if (!QDir().exists(configDir)) {
        if (!QDir().mkpath(configDir)) {
            qCritical() << "Could not create configuration directory:" << configDir;
            return;
        }
    }

    m_configFilePath = configDir + "/config.json";
}

void ConfigService::readConfig() {
    if (QFile::exists(m_configFilePath)) {
        QFile file(m_configFilePath);

        if (!file.open(QIODevice::ReadOnly)) {
            qInfo() << "Configuration file not found. Creating a new one with default values.";
        } else {
            QByteArray data = file.readAll();
            file.close();

            if (!data.isEmpty()) {
                QJsonParseError parseError;
                QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
                if (parseError.error != QJsonParseError::NoError) {
                    qCritical() << "Error parsing config.json:" << parseError.errorString()
                                << ". Will reset to default.";
                    m_root = QJsonObject();
                } else {
                    m_root = doc.object();
                }
            }
        }
    }

    // Ensure the root is a valid object.
    if (m_root.isEmpty()) {
        m_root = QJsonObject();
    }

    // Now, populate the in-memory server cache from the JSON object.
    m_servers.clear();
    if (m_root.contains("servers") && m_root["servers"].isArray()) {
        const QJsonArray serversArray = m_root["servers"].toArray();
        for (const auto& entry : serversArray) {
            if (entry.isString()) {
                QString server = entry.toString();
                if (!m_servers.contains(server)) {
                    m_servers.append(server);
                }
            }
        }
    }
}

QJsonObject& ConfigService::getRoot() {
    return m_root;
}

const QJsonObject& ConfigService::getRoot() const {
    return m_root;
}

bool ConfigService::isFirstInstance() const {
    return m_isFirstInstance;
}

QStringList ConfigService::servers() const {
    return m_servers;
}

bool ConfigService::addServer(const QString& server) {
    if (server.isEmpty() || m_servers.contains(server)) {
        return false;
    }

    m_servers.append(server);
    emit serversChanged();
    return true;
}

void ConfigService::removeServer(const QString& server) {
    if (m_servers.removeAll(server) > 0) {
        emit serversChanged();
    }
}

} // namespace qt_client
