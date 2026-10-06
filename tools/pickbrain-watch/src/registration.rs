use super::{options, Options};

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(any(target_os = "macos", test))]
const LABEL: &str = "com.dropbox.pickbrain";

#[derive(Serialize, Deserialize)]
struct Configuration {
    directory: PathBuf,
    environment: BTreeMap<String, OsString>,
    log: PathBuf,
}

fn write_private(path: &Path, contents: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().context("configuration has no parent directory")?)?;
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).with_context(|| format!("write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(contents)?;
    Ok(())
}

pub(super) fn restore(path: &Path) -> Result<()> {
    let configuration: Configuration = serde_json::from_slice(&fs::read(path)
        .with_context(|| format!("read {}", path.display()))?)?;
    for (key, value) in configuration.environment {
        env::set_var(key, value);
    }
    let log = OpenOptions::new().create(true).append(true).open(&configuration.log)
        .with_context(|| format!("open {}", configuration.log.display()))?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        // Keep errors from both the supervisor and its children in the same log.
        for descriptor in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
            if unsafe { libc::dup2(log.as_raw_fd(), descriptor) } == -1 {
                return Err(std::io::Error::last_os_error()).context("redirect watcher output");
            }
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::IntoRawHandle;
        use windows_sys::Win32::System::Console::{FreeConsole, SetStdHandle, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
        // A logon task should not leave a console window open. Manual --watch
        // runs keep their terminal because they do not load a service configuration.
        unsafe { FreeConsole(); }
        let handle = log.into_raw_handle();
        // The process owns this handle until exit; child processes inherit it.
        for descriptor in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            if unsafe { SetStdHandle(descriptor, handle) } == 0 {
                return Err(std::io::Error::last_os_error()).context("redirect watcher output");
            }
        }
    }
    env::set_current_dir(&configuration.directory)
        .with_context(|| format!("enter {}", configuration.directory.display()))?;
    Ok(())
}

#[cfg(any(target_os = "macos", windows, test))]
fn xml(value: &str) -> String {
    value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
        .replace('"', "&quot;").replace('\'', "&apos;")
}

fn arguments(options: &Options, config: &Path) -> Result<Vec<String>> {
    Ok(vec![options.pickbrain.to_str().context("executable path is not Unicode")?.to_owned(),
        "--watch".into(), "--delay".into(), options.delay.as_secs().to_string(),
        "--max-delay".into(), options.max_delay.as_secs().to_string(),
        "--service-config".into(), config.to_str().context("configuration path is not Unicode")?.to_owned()])
}

#[cfg(any(target_os = "macos", test))]
fn launch_agent(args: &[String], log: &Path) -> String {
    let args: String = args.iter().map(|arg| format!("<string>{}</string>\n", xml(arg))).collect();
    let log = xml(&log.to_string_lossy());
    format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
        <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
        <plist version=\"1.0\"><dict>\n\
        <key>Label</key><string>{LABEL}</string>\n\
        <key>ProgramArguments</key><array>{args}</array>\n\
        <key>RunAtLoad</key><true/>\n\
        <key>KeepAlive</key><true/>\n\
        <key>StandardOutPath</key><string>{log}</string>\n\
        <key>StandardErrorPath</key><string>{log}</string>\n\
        </dict></plist>\n")
}

#[cfg(any(target_os = "linux", test))]
fn systemd_service(args: &[String]) -> String {
    // systemd expands percent specifiers and dollar variables even inside quotes.
    let args = args.iter().map(|arg| format!("\"{}\"", arg.replace('\\', "\\\\")
        .replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r")
        .replace('%', "%%").replace('$', "$$"))).collect::<Vec<_>>().join(" ");
    format!("[Unit]\nDescription=Pickbrain session watcher\n\n\
        [Service]\nExecStart={args}\nRestart=always\n\n\
        [Install]\nWantedBy=default.target\n")
}

#[cfg(any(windows, test))]
fn windows_argument(value: &str) -> String {
    // CommandLineToArgvW quoting, including trailing backslashes before the quote.
    let mut result = String::from("\"");
    let mut slashes = 0;
    for character in value.chars() {
        if character == '\\' { slashes += 1; continue; }
        result.extend(std::iter::repeat_n('\\', if character == '"' { slashes * 2 + 1 } else { slashes }));
        result.push(character);
        slashes = 0;
    }
    result.extend(std::iter::repeat_n('\\', slashes * 2));
    result.push('"');
    result
}

#[cfg(any(windows, test))]
fn scheduled_task(args: &[String], sid: &str) -> String {
    let command = xml(&args[0]);
    let arguments = xml(&args[1..].iter().map(|arg| windows_argument(arg)).collect::<Vec<_>>().join(" "));
    let sid = xml(sid);
    format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
        <Task version=\"1.2\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\n\
        <Triggers><LogonTrigger><Enabled>true</Enabled><UserId>{sid}</UserId></LogonTrigger></Triggers>\n\
        <Principals><Principal id=\"User\"><UserId>{sid}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>\n\
        <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n\
        <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n\
        <ExecutionTimeLimit>PT0S</ExecutionTimeLimit><Enabled>true</Enabled></Settings>\n\
        <Actions Context=\"User\"><Exec><Command>{command}</Command><Arguments>{arguments}</Arguments></Exec></Actions>\n\
        </Task>\n")
}

fn checked(command: &mut Command) -> Result<Output> {
    let output = command.output().with_context(|| format!("run {command:?}"))?;
    if !output.status.success() {
        bail!("{command:?} failed ({}): {}{}", output.status,
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn install(home: &Path, state: &Path, args: &[String]) -> Result<()> {
    let path = home.join(format!("Library/LaunchAgents/{LABEL}.plist"));
    let domain = format!("gui/{}", unsafe { libc::geteuid() });
    let service = format!("{domain}/{LABEL}");
    // Only ignore a missing job; a failed bootout of a loaded job must stop the update.
    if Command::new("/bin/launchctl").args(["print", &service]).output()?.status.success() {
        checked(Command::new("/bin/launchctl").args(["bootout", &service]))?;
    }
    write_private(&path, launch_agent(args, &state.join("watch.log")).as_bytes())?;
    checked(Command::new("/bin/launchctl").args(["enable", &service]))?;
    checked(Command::new("/bin/launchctl").arg("bootstrap").arg(&domain).arg(&path))?;
    eprintln!("Registered {service}; log: {}", state.join("watch.log").display());
    Ok(())
}

#[cfg(target_os = "linux")]
fn install(home: &Path, state: &Path, args: &[String]) -> Result<()> {
    let config = env::var_os("XDG_CONFIG_HOME").filter(|path| !path.is_empty())
        .map(PathBuf::from).unwrap_or_else(|| home.join(".config"));
    let path = config.join("systemd/user/pickbrain.service");
    write_private(&path, systemd_service(args).as_bytes())?;
    checked(Command::new("systemctl").args(["--user", "daemon-reload"]))?;
    checked(Command::new("systemctl").args(["--user", "enable", "pickbrain.service"]))?;
    checked(Command::new("systemctl").args(["--user", "restart", "pickbrain.service"]))?;
    eprintln!("Registered pickbrain.service; log: {}", state.join("watch.log").display());
    Ok(())
}

#[cfg(windows)]
fn install(_home: &Path, state: &Path, args: &[String]) -> Result<()> {
    let identity = checked(Command::new("whoami.exe").args(["/user", "/fo", "csv", "/nh"]))?;
    let text = String::from_utf8(identity.stdout)?;
    let sid = text.trim().rsplit(',').next().context("read current user SID")?.trim_matches('"');
    anyhow::ensure!(sid.starts_with("S-1-") && sid.chars().all(|c| c.is_ascii_digit() || c == '-' || c == 'S'), "invalid current user SID");
    let name = format!("Pickbrain-{sid}");
    let path = state.join("watch-task.xml");
    write_private(&path, scheduled_task(args, sid).as_bytes())?;
    if Command::new("schtasks.exe").args(["/Query", "/TN", &name]).output()?.status.success() {
        // /End reports an error when an existing task is already stopped.
        let _ = Command::new("schtasks.exe").args(["/End", "/TN", &name]).output()?;
    }
    checked(Command::new("schtasks.exe").args(["/Create", "/TN", &name, "/XML"]).arg(&path).arg("/F"))?;
    checked(Command::new("schtasks.exe").args(["/Run", "/TN", &name]))?;
    eprintln!("Registered {name}; log: {}", state.join("watch.log").display());
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn install(_home: &Path, _state: &Path, _args: &[String]) -> Result<()> {
    bail!("automatic registration is supported on macOS, Linux with systemd, and Windows")
}

pub fn register(args: impl Iterator<Item = OsString>) -> Result<()> {
    let Some(options) = options(args)? else { return Ok(()); };
    anyhow::ensure!(options.service_config.is_none(), "--service-config is only used by registered watchers");
    let home = env::var_os("HOME").filter(|home| !home.is_empty())
        .or_else(|| env::var_os("USERPROFILE")).context("HOME or USERPROFILE must be set")?;
    let home = PathBuf::from(home).canonicalize().context("locate home directory")?;
    // Keep the registration configuration at a fixed location so changing
    // PICKBRAIN_DIR on a later registration cannot leave a second service behind.
    let state = home.join(".pickbrain");
    let configuration = Configuration {
        directory: env::current_dir()?,
        environment: ["HOME", "USERPROFILE", "PATH", "PICKBRAIN_DIR", "EXTRA_CLAUDE_DIRS",
            "EXTRA_CODEX_DIRS", "EXTRA_PI_DIRS", "WARP_ASSETS", "PRE_INGEST_COMMAND"]
            .into_iter().filter_map(|key| env::var_os(key).map(|value| (key.to_owned(), value))).collect(),
        log: state.join("watch.log"),
    };
    let path = state.join("watch.json");
    write_private(&path, &serde_json::to_vec_pretty(&configuration)?)?;
    install(&home, &state, &arguments(&options, &path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_watcher_restores_environment_directory_and_child_logging() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("working directory");
        fs::create_dir(&directory).unwrap();
        let path = root.path().join("watch.json");
        let log = root.path().join("watch.log");
        let configuration = Configuration {
            directory,
            environment: BTreeMap::from([("PICKBRAIN_REGISTRATION_TEST_VALUE".into(), "spaces & quotes \" $HOME".into())]),
            log: log.clone(),
        };
        write_private(&path, &serde_json::to_vec(&configuration).unwrap()).unwrap();
        let status = Command::new(env::current_exe().unwrap())
            .args(["--ignored", "--exact", "registration::tests::registered_process_entry", "--nocapture"])
            .env("PICKBRAIN_REGISTRATION_TEST_CONFIG", &path)
            .env_remove("PICKBRAIN_REGISTRATION_TEST_CHILD")
            .output().unwrap();
        assert!(status.status.success(), "{}", String::from_utf8_lossy(&status.stderr));
        let output = fs::read_to_string(log).unwrap();
        assert!(output.contains("supervisor log"));
        assert!(output.contains("child log"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }

    #[test]
    #[ignore = "subprocess entry point for registration tests"]
    fn registered_process_entry() {
        if env::var_os("PICKBRAIN_REGISTRATION_TEST_CHILD").is_some() {
            eprintln!("child log");
            return;
        }
        let path = PathBuf::from(env::var_os("PICKBRAIN_REGISTRATION_TEST_CONFIG").unwrap());
        restore(&path).unwrap();
        assert_eq!(env::var("PICKBRAIN_REGISTRATION_TEST_VALUE").unwrap(), "spaces & quotes \" $HOME");
        assert_eq!(env::current_dir().unwrap().file_name().unwrap(), "working directory");
        eprintln!("supervisor log");
        assert!(Command::new(env::current_exe().unwrap())
            .args(["--ignored", "--exact", "registration::tests::registered_process_entry", "--nocapture"])
            .env("PICKBRAIN_REGISTRATION_TEST_CHILD", "1").status().unwrap().success());
    }

    #[test]
    fn service_definitions_preserve_paths_and_do_not_expand_shell_syntax() {
        let args = vec!["/a b/&<\"pickbrain".into(), "--watch".into(), "/state/%home/$USER.json".into()];
        let plist = launch_agent(&args, Path::new("/log & space/watch.log"));
        assert!(plist.contains("/a b/&amp;&lt;&quot;pickbrain"));
        assert!(plist.contains("<key>KeepAlive</key><true/>"));
        let unit = systemd_service(&args);
        assert!(unit.contains("\"/a b/&<\\\"pickbrain\""));
        assert!(unit.contains("/state/%%home/$$USER.json"));
        let task = scheduled_task(&args, "S-1-5-21-123");
        assert!(task.contains("<LogonType>InteractiveToken</LogonType>"));
        assert!(task.contains("<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>"));
        assert!(task.contains("<UserId>S-1-5-21-123</UserId>"));
        assert_eq!(windows_argument("C:\\a b\\"), "\"C:\\a b\\\\\"");
    }
}
