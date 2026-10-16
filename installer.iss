#define AppName "CLPT FSDT RK"
#define AppVersion "1.0.0"
#define AppExeName "CLPT_FSDT_RK.exe"

[Setup]
AppId={{8D9BE3FC-5FEB-4CB3-99D2-E2E16F226C0C}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\CLPT FSDT RK
DefaultGroupName=CLPT FSDT RK
OutputDir=installer_output
OutputBaseFilename=CLPT_FSDT_RK_Setup_Windows_x64
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayName={#AppName}

[Files]
Source: "dist\CLPT_FSDT_RK\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\CLPT FSDT RK"; Filename: "{app}\{#AppExeName}"
Name: "{userdesktop}\CLPT FSDT RK"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Run CLPT FSDT RK"; Flags: nowait postinstall skipifsilent
