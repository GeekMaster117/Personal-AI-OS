app_blacklist: set[str] = {
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
    "searchhost.exe"
}

title_blacklist: set[str] = {
    "desktopwindowxamlsource",
    "chrome legacy window",
    "default ime"
}

specific_title_blacklist: dict[str, set[str]] = {
    "explorer.exe": {
        "running applications",
        "program manager"
    }
}