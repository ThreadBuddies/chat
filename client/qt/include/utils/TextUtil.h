#pragma once

#include <QString>
#include <QValidator>

namespace qt_client {
namespace TextUtil {

/**
 * @brief Sanitizes a user-facing identifier (username, room name).
 *
 * Strips C0/C1 Unicode control characters, trims surrounding whitespace and
 * requires the result to contain at least one alphanumeric Unicode character.
 *
 * @param input The raw string to sanitize.
 * @return The cleaned string on success, or a null QString if the input is
 *         empty after stripping/trimming or contains no alphanumeric character.
 */
QString sanitizeInput(const QString& input);

/**
 * @brief Validates a chat message before sending.
 *
 * Trims whitespace from the edges and rejects empty or whitespace-only input.
 * Internal newlines, tabs, emoji and other Unicode content are preserved —
 * messages are free-form. A defensive length check enforces
 * common::limits::MAX_MESSAGE_LENGTH; CodePointValidator already enforces the
 * same limit at the input layer.
 *
 * @param input The raw message text from the input field.
 * @return The trimmed message on success, or a null QString if the message is
 *         empty/whitespace-only or exceeds MAX_MESSAGE_LENGTH code points.
 */
QString validateMessage(const QString& input);

/**
 * @brief Counts Unicode code points in a string.
 *
 * Surrogate pairs are counted as a single code point.
 *
 * @param text The string to measure.
 * @return The number of Unicode code points.
 */
int countMessageCodePoints(const QString& text);

/**
 * @brief Finds the previous grapheme-cluster boundary in a string.
 *
 * Walks backwards from fromIndex and returns the UTF-16 index of the previous
 * grapheme-cluster boundary strictly before it. Handles emoji ZWJ sequences,
 * skin-tone modifiers, regional indicator flags and variation selectors as
 * single clusters — used by QML to delete a whole emoji on Backspace.
 *
 * @param text The string to scan.
 * @param fromIndex The UTF-16 index to start scanning from.
 * @return The boundary index, or fromIndex if it is already at or before zero.
 */
int previousGraphemeBoundary(const QString& text, int fromIndex);

} // namespace TextUtil

/**
 * @brief QValidator rejecting input that exceeds a code-point limit.
 *
 * Used as TextField.validator in QML to block over-limit edits at the input
 * layer (typing, paste, drag-drop, IME) without modifying the text. For
 * controls where QValidator cannot be attached directly (TextEdit/TextArea),
 * QML can query @ref isOverLimit and revert manually.
 */
class CodePointValidator : public QValidator {
    Q_OBJECT
    Q_PROPERTY(int maxCodePoints READ maxCodePoints WRITE setMaxCodePoints
               NOTIFY maxCodePointsChanged)
public:
    using QValidator::QValidator;

    State validate(QString& input, int& pos) const override;

    /**
     * @brief Checks whether text exceeds maxCodePoints.
     *
     * Exposed to QML so components without a validator slot (TextEdit/TextArea)
     * can apply the same length check the validator uses.
     *
     * @param text The string to check.
     * @return true if text exceeds maxCodePoints, false otherwise (also false
     *         when maxCodePoints is unset / non-positive).
     */
    Q_INVOKABLE bool isOverLimit(const QString& text) const;

    int maxCodePoints() const { return m_max; }
    void setMaxCodePoints(int v);

signals:
    void maxCodePointsChanged();

private:
    int m_max = -1;
};

} // namespace qt_client
