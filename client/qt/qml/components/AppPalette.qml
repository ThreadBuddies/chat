pragma Singleton
import QtQuick

// Central colour palette — access directly as AppPalette.accent etc. (singleton, no instantiation needed)
QtObject {

    // ── Backgrounds ───────────────────────────────────────────────────────────────────────
    readonly property color bgBase:              "#ffffff"   // main page background
    readonly property color bgSurface:           "#f7f7f8"   // sidebars, panels
    readonly property color bgSurfaceAlt:        "#f0f0f2"   // inset / deeper surface
    readonly property color bgHover:             "#eaeaed"   // generic hover state
    readonly property color bgSelected:          "#e0ecfa"   // selected list item

    // ── Accent (blue) ─────────────────────────────────────────────────────────────────────
    readonly property color accent:              "#2563EB"   // blue-600 — primary action
    readonly property color accentLight:         "#eff6ff"   // blue-50  — subtle fill
    readonly property color accentMid:           "#93c5fd"   // blue-300 — scrollbar hover

    // ── Text ──────────────────────────────────────────────────────────────────────────────
    readonly property color textPrimary:         "#18181b"
    readonly property color textSecondary:       "#71717a"
    readonly property color textMuted:           "#a1a1aa"

    // ── Borders ───────────────────────────────────────────────────────────────────────────
    readonly property color borderColor:         "#e4e4e7"

    // ── Error ─────────────────────────────────────────────────────────────────────────────
    readonly property color colorError:          "#dc2626"
    readonly property color colorErrorBg:        "#fee2e2"
    readonly property color colorErrorBorder:    "#fca5a5"

    // ── Success ───────────────────────────────────────────────────────────────────────────
    readonly property color colorSuccess:        "#16a34a"
    readonly property color colorSuccessBg:      "#dcfce7"
    readonly property color colorSuccessBorder:  "#86efac"

    // ── Warning ───────────────────────────────────────────────────────────────────────────
    readonly property color colorWarn:           "#d97706"
    readonly property color colorWarnBg:         "#fef3c7"
    readonly property color colorWarnBorder:     "#fcd34d"
}
