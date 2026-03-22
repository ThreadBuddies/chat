#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "controllers/app_controller.h"
#include "services/config_service.h"
#include "services/ws_service.h"

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    app.setApplicationName("SlightlyPrettyChat");
    app.setOrganizationName("ThreadBuddies");

    qt_client::ConfigService configService(app.applicationName());
    qt_client::WebSocketService wsService;
    qt_client::AppController controller(&configService, &wsService);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("appController", &controller);
    engine.rootContext()->setContextProperty("configService", &configService);

    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));

    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
