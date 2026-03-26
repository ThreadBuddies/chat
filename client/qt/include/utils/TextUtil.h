#pragma once

#include <QString>

namespace qt_client {
namespace TextUtil {

/// Trims whitespace and validates that the string contains at least one
/// alphanumeric Unicode character — mirrors the wx client's SanitizeInput.
/// Returns the trimmed string on success, or a null QString on failure.
QString sanitizeInput(const QString& input);

} // namespace TextUtil
} // namespace qt_client
