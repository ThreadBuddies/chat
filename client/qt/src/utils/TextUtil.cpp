#include <utils/TextUtil.h>
#include <common/utils/limits.h>
#include <QList>
#include <QTextBoundaryFinder>

namespace qt_client {
namespace TextUtil {

QString sanitizeInput(const QString& input) {
    // 1. Strip C0/C1 control characters (\n, \t, \r, etc.) — identifiers
    //    must not contain these.
    QString stripped;
    stripped.reserve(input.size());
    for (const QChar& ch : input) {
        if (ch.category() != QChar::Other_Control) {
            stripped.append(ch);
        }
    }

    // 2. Trim whitespace from edges.
    QString trimmed = stripped.trimmed();
    if (trimmed.isEmpty()) {
        return {};
    }

    // 3. Require at least one alphanumeric character.
    for (const QChar& ch : trimmed) {
        if (ch.isLetterOrNumber()) {
            return trimmed;
        }
    }

    return {};
}

int countMessageCodePoints(const QString& text) {
    return static_cast<int>(text.toUcs4().size());
}

int previousGraphemeBoundary(const QString& text, int fromIndex) {
    if (fromIndex <= 0 || text.isEmpty()) {
        return fromIndex;
    }

    const int clamped = qMin(fromIndex, text.size());
    QTextBoundaryFinder finder(QTextBoundaryFinder::Grapheme, text);
    finder.setPosition(clamped);
    const int prev = finder.toPreviousBoundary();
    if (prev < 0) {
        return 0;
    }
    return prev;
}

QString validateMessage(const QString& input) {
    // 1. Trim whitespace from edges. Internal newlines, tabs and emoji are
    //    preserved — messages are free-form content.
    QString text = input.trimmed();

    // 2. Require at least one visible (non-whitespace) code unit.
    //    Surrogate pairs (emoji etc.) are never isSpace(), so they pass.
    bool hasVisible = false;
    for (const QChar& ch : text) {
        if (!ch.isSpace()) {
            hasVisible = true;
            break;
        }
    }
    if (!hasVisible) {
        return {};
    }

    // 3. Defensive length check; CodePointValidator enforces the same limit
    //    at the input layer.
    if (countMessageCodePoints(text) > static_cast<int>(common::limits::MAX_MESSAGE_LENGTH)) {
        return {};
    }

    return text;
}

} // namespace TextUtil

QValidator::State CodePointValidator::validate(QString& input, int& /*pos*/) const {
    if (isOverLimit(input)) {
        return Invalid;
    }
    return Acceptable;
}

bool CodePointValidator::isOverLimit(const QString& text) const {
    return m_max > 0 && TextUtil::countMessageCodePoints(text) > m_max;
}

void CodePointValidator::setMaxCodePoints(int v) {
    if (v == m_max) return;
    m_max = v;
    emit maxCodePointsChanged();
}

} // namespace qt_client
