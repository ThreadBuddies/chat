#include <utils/text_util.h>

namespace qt_client {
namespace TextUtil {

QString sanitizeInput(const QString& input) {
    QString trimmed = input.trimmed();
    if (trimmed.isEmpty())
        return {};

    for (const QChar& ch : trimmed) {
        if (ch.isLetterOrNumber())
            return trimmed;
    }

    // No alphanumeric character found.
    return {};
}

} // namespace TextUtil
} // namespace qt_client
