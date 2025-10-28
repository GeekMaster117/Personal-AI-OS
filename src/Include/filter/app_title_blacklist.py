app_blacklist: frozenset[str] = frozenset([
    "textinputhost.exe",
    "applicationframehost.exe",
    "dllhost.exe",
    "sihost.exe",
    "searchui.exe",
    "ctfmon.exe",
    "runtimebroker.exe",
    "startmenuexperiencehost.exe",
    "systemsettings.exe",
    "systemhost.exe",
    "searchhost.exe",
    "video.ui.exe"
])

title_blacklist: frozenset[str] = frozenset([
    "desktopwindowxamlsource",
    "chrome legacy window",
    "default ime",
    "popuphost"
    "ok",
    "cancel",
    "ok, don't show again"
])

specific_title_blacklist: dict[str, frozenset[str]] = {
    "explorer.exe": frozenset([
        "running applications",
        "program manager"
    ])
}