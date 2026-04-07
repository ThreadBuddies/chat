#pragma once

#include <QString>

namespace qt_client {
namespace TextUtil {

/// Trims whitespace and validates that the string contains at least one
/// alphanumeric Unicode character — mirrors the wx client's SanitizeInput.
/// Returns the trimmed string on success, or a null QString on failure.
QString sanitizeInput(const QString& input);

/// NFC-normalises, strips control characters, trims, and validates a chat
/// message. Returns the normalised text on success, or a null QString if the
/// message is empty/whitespace-only or exceeds MAX_MESSAGE_LENGTH (512)
/// Unicode code points.
QString validateMessage(const QString& input);

/// Returns the number of Unicode code points in text (surrogate pairs counted
/// as one). Equivalent to utf8::distance on the server side.
int countMessageCodePoints(const QString& text);

} // namespace TextUtil
} // namespace qt_client
