#include <utils/TextUtil.h>
#include <common/utils/limits.h>
#include <QList>

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

int countMessageCodePoints(const QString& text) {
    return static_cast<int>(text.toUcs4().size());
}

QString validateMessage(const QString& input) {
    // 1. NFC normalisation
    QString text = input.normalized(QString::NormalizationForm_C);

    // 2. Remove problematic characters (C0/C1 Unicode control characters)
    QString cleaned;
    cleaned.reserve(text.size());
    for (const QChar& ch : text) {
        if (ch.category() != QChar::Other_Control)
            cleaned.append(ch);
    }

    // 3. Trim
    text = cleaned.trimmed();

    // 4. Require at least one visible (non-whitespace) code unit
    //    Surrogate pairs (emoji etc.) are never isSpace(), so they pass.
    bool hasVisible = false;
    for (const QChar& ch : text) {
        if (!ch.isSpace()) {
            hasVisible = true;
            break;
        }
    }
    if (!hasVisible)
        return {};

    // 5. Enforce MAX_MESSAGE_LENGTH Unicode code points
    if (countMessageCodePoints(text) > static_cast<int>(common::limits::MAX_MESSAGE_LENGTH))
        return {};

    // 6. Return normalised text
    return text;
}

} // namespace TextUtil
} // namespace qt_client
