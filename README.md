# SEINXCAL Calendar App with Voice Input

A modern calendar application for Windows with Google Calendar integration and voice input (powered by OpenAI Whisper).

## Features
- Google Calendar sync (view, add, update, delete events)
- Voice input for event fields (supports English and Japanese)
- Search events by date
- Quick reset to today
- Light/Dark theme
- Multi-language UI (English/Japanese)
- Auto-refresh events

## Requirements
- Windows 10/11
- Python 3.8+
- Microphone (for voice input)
- [ffmpeg](https://ffmpeg.org/) in PATH (for audio processing)
- Google API credentials (`credentials.json`)

## Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your `credentials.json` (Google API) in the app directory.
4. Run the app:
   ```bash
   python claudever.py
   ```

## Packaging as an EXE
1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Build the executable:
   ```bash
   pyinstaller --noconfirm --onedir --windowed --icon=icons/calendar-app-50.png claudever.py
   ```
   - The EXE will be in the `dist/` folder.
   - Copy `credentials.json`, `icons/`, and any other required files to the same folder as the EXE.

## Creating an Installer (Inno Setup)
1. Download and install [Inno Setup](https://jrsoftware.org/isinfo.php).
2. Use the provided `innosetup.iss` script (see below) as a template.
3. Build the installer in Inno Setup.

## Inno Setup Script Example
```
[Setup]
AppName=SEINXCAL Calendar
AppVersion=1.0
DefaultDirName={pf}\SEINXCAL
DefaultGroupName=SEINXCAL
OutputDir=dist
OutputBaseFilename=SEINXCAL_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\claudever.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "credentials.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "icons\*"; DestDir: "{app}\icons"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SEINXCAL Calendar"; Filename: "{app}\claudever.exe"

[Run]
Filename: "{app}\claudever.exe"; Description: "Launch SEINXCAL Calendar"; Flags: nowait postinstall skipifsilent
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Credits
- OpenAI Whisper
- PyQt5
- Google Calendar API
- qtawesome
- scipy, numpy, sounddevice, tzlocal 