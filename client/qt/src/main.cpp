#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>

#include <controllers/AppController.h>
#include <services/ConfigService.h>
#include <services/WsService.h>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Fusion");
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
