#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QFontDatabase>
#include <QDir>

#include <controllers/AppController.h>
#include <services/ConfigService.h>
#include <services/WsService.h>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Fusion");
    app.setApplicationName("SlightlyPrettyChat");
    app.setOrganizationName("ThreadBuddies");

#ifdef Q_OS_MACOS
    // On macOS the executable lives in Contents/MacOS/, fonts in Contents/Resources/fonts/
    const QString fontDir = app.applicationDirPath() + "/../Resources/fonts";
#else
    const QString fontDir = app.applicationDirPath() + "/fonts";
#endif
    for (const QString& file : QDir(fontDir).entryList({"*.ttf"}, QDir::Files))
        QFontDatabase::addApplicationFont(fontDir + "/" + file);

    QFont appFont;
    appFont.setFamilies({"Roboto", "Noto Color Emoji", "Noto Sans SC"});
    QGuiApplication::setFont(appFont);

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
