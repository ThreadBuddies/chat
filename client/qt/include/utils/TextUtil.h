#pragma once

#include <QString>
#include <QValidator>

namespace qt_client {
namespace TextUtil {

/// Strips control characters, trims whitespace, and requires at least one
/// alphanumeric Unicode character. Used for user-facing identifiers like
/// usernames and room names. Returns the cleaned string on success, or a
/// null QString on failure.
QString sanitizeInput(const QString& input);

/// Trims whitespace from the edges and validates a chat message. Preserves
/// internal newlines, tabs, emoji and other Unicode content — messages are
/// free-form. Returns the trimmed text on success, or a null QString if the
/// message is empty/whitespace-only or exceeds MAX_MESSAGE_LENGTH (512)
/// Unicode code points.
QString validateMessage(const QString& input);

/// Returns the number of Unicode code points in text (surrogate pairs counted
/// as one). Equivalent to utf8::distance on the server side.
int countMessageCodePoints(const QString& text);

/// Returns the UTF-16 index of the previous grapheme-cluster boundary strictly
/// before fromIndex. Handles emoji ZWJ sequences, skin-tone modifiers, regional
/// indicator flags and variation selectors as single clusters. Returns
/// fromIndex if it is already at or before zero.
int previousGraphemeBoundary(const QString& text, int fromIndex);

} // namespace TextUtil

/// QValidator that rejects any input exceeding maxCodePoints Unicode code
/// points. Used as TextField.validator in QML to block over-limit edits at the
/// input layer (typing, paste, drag-drop, IME) without modifying the text.
class CodePointValidator : public QValidator {
    Q_OBJECT
    Q_PROPERTY(int maxCodePoints READ maxCodePoints WRITE setMaxCodePoints
               NOTIFY maxCodePointsChanged)
public:
    using QValidator::QValidator;

    State validate(QString& input, int& pos) const override;

    /// True if text exceeds maxCodePoints. Lets QML query the same length-check
    /// the validator uses, for components where QValidator can't be attached
    /// directly (TextEdit/TextArea).
    Q_INVOKABLE bool isOverLimit(const QString& text) const;

    int maxCodePoints() const { return m_max; }
    void setMaxCodePoints(int v);

signals:
    void maxCodePointsChanged();

private:
    int m_max = -1;
};

} // namespace qt_client
