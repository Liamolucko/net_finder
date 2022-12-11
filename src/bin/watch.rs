use std::cmp::Reverse;
use std::fmt::Write;
use std::iter::zip;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::{fs, io};

use clap::Parser;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::ClearType;
use crossterm::{cursor, terminal, ExecutableCommand};
use net_finder::{ColoredNet, Pos, State};
use notify::{RecursiveMode, Watcher};

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let Args { path } = Args::parse();

    let changed = Arc::new(AtomicBool::new(false));
    let mut watcher = notify::recommended_watcher({
        let changed = Arc::clone(&changed);
        move |res| match res {
            Ok(_) => changed.store(true, Ordering::Relaxed),
            // ignore it? idk
            Err(_) => {}
        }
    })?;
    watcher.watch(&path, RecursiveMode::NonRecursive)?;

    terminal::enable_raw_mode()?;
    io::stdout().execute(terminal::EnterAlternateScreen)?;
    let result = main_loop(path, changed);
    io::stdout().execute(terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Mode {
    /// Show the cuboids in the fixed order they show up in in the file.
    Fixed,
    /// Show the cuboids ordered from smallest area to largest area.
    Ascending,
    /// Show the cuboids ordered from largest area to smallest area.
    Descending,
}

fn main_loop(path: PathBuf, changed: Arc<AtomicBool>) -> anyhow::Result<()> {
    let mut mode = Mode::Fixed;
    let mut contents = render_contents(&path, mode)?;
    let mut scroll: usize = 0;
    let mut needs_render = true;
    loop {
        if crossterm::event::poll(Duration::from_millis(100))? {
            match crossterm::event::read()? {
                Event::Key(event) if event.kind == KeyEventKind::Press => match event.code {
                    KeyCode::Esc | KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('c' | 'd') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(())
                    }
                    KeyCode::Char('f') => {
                        mode = Mode::Fixed;
                        changed.store(true, Ordering::Relaxed);
                    }
                    KeyCode::Char('a') => {
                        mode = Mode::Ascending;
                        changed.store(true, Ordering::Relaxed);
                    }
                    KeyCode::Char('d') => {
                        mode = Mode::Descending;
                        changed.store(true, Ordering::Relaxed);
                    }
                    KeyCode::Up => {
                        scroll = scroll.saturating_sub(1);
                        needs_render = true;
                    }
                    KeyCode::Down => {
                        scroll = scroll.saturating_add(1);
                        needs_render = true;
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        if changed.swap(false, Ordering::Relaxed) {
            contents = render_contents(&path, mode)?;
            needs_render = true;
        }

        if needs_render {
            render(&contents, scroll)?;
            needs_render = false;
        }
    }
}

fn render_contents(path: &Path, mode: Mode) -> anyhow::Result<String> {
    // Retry until we read a valid state, since if it fails we've most likely tried
    // to read the file while it's still being written to.
    let mut state: State = loop {
        let bytes = fs::read(path)?;
        if let Ok(state) = postcard::from_bytes(&bytes) {
            break state;
        }
    };
    match mode {
        Mode::Fixed => {}
        Mode::Ascending => state.finders.sort_by_key(|finder| finder.area),
        Mode::Descending => state.finders.sort_by_key(|finder| Reverse(finder.area)),
    }
    let mut result = String::new();
    for (i, finder) in state.finders.into_iter().enumerate() {
        // We show one net for each cuboid.
        let mut nets =
            vec![ColoredNet::new(finder.net.width(), finder.net.height()); finder.cuboids().len()];
        for x in 0..finder.net.width() {
            for y in 0..finder.net.height() {
                let pos = Pos::new(x, y);
                if finder.net.filled(pos) {
                    zip(&mut nets, &finder.pos_possibilities[&pos][0])
                        .for_each(|(net, face_pos)| net.set(pos, face_pos.face))
                }
            }
        }

        let strings = nets.into_iter().map(|net| net.to_string()).collect();
        if i != 0 {
            result.push('\n');
        }
        writeln!(result, "Area: {}/{}", finder.area, finder.target_area).unwrap();
        result.push_str(&join_horizontal(strings));
    }
    Ok(result)
}

fn join_horizontal(strings: Vec<String>) -> String {
    let mut lines: Vec<_> = strings.iter().map(|s| s.lines()).collect();
    let mut out = String::new();
    loop {
        for (i, iter) in lines.iter_mut().enumerate() {
            if i != 0 {
                out += " ";
            }
            if let Some(line) = iter.next() {
                out += line;
            } else {
                return out;
            }
        }
        out += "\n";
    }
}

fn render(contents: &str, scroll: usize) -> anyhow::Result<()> {
    io::stdout().execute(terminal::Clear(ClearType::All))?;
    io::stdout().execute(cursor::MoveTo(0, 0))?;
    for line in contents
        .lines()
        .skip(scroll)
        .take((terminal::size()?.1 - 1).into())
    {
        print!("{line}\r\n");
    }
    Ok(())
}
